import Mathlib

namespace NUMINAMATH_CALUDE_tim_weekly_earnings_l1778_177884

/-- Calculates Tim's total weekly earnings including bonuses -/
def timWeeklyEarnings (tasksPerDay : ℕ) (workDaysPerWeek : ℕ) 
  (tasksPay1 tasksPay2 tasksPay3 : ℕ) (rate1 rate2 rate3 : ℚ) 
  (bonusThreshold : ℕ) (bonusAmount : ℚ) 
  (performanceBonusThreshold : ℕ) (performanceBonusAmount : ℚ) : ℚ :=
  let dailyEarnings := tasksPay1 * rate1 + tasksPay2 * rate2 + tasksPay3 * rate3
  let dailyBonuses := (tasksPerDay / bonusThreshold) * bonusAmount
  let weeklyEarnings := (dailyEarnings + dailyBonuses) * workDaysPerWeek
  let performanceBonus := if tasksPerDay ≥ performanceBonusThreshold then performanceBonusAmount else 0
  weeklyEarnings + performanceBonus

/-- Tim's total weekly earnings are $1058 -/
theorem tim_weekly_earnings :
  timWeeklyEarnings 100 6 40 30 30 (6/5) (3/2) 2 50 10 90 20 = 1058 := by
  sorry

end NUMINAMATH_CALUDE_tim_weekly_earnings_l1778_177884


namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l1778_177866

theorem fly_distance_from_ceiling (x y z : ℝ) :
  x = 2 ∧ y = 6 ∧ Real.sqrt (x^2 + y^2 + z^2) = 10 → z = 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l1778_177866


namespace NUMINAMATH_CALUDE_simplify_polynomial_l1778_177838

theorem simplify_polynomial (y : ℝ) : y * (4 * y^2 + 3) - 6 * (y^2 + 3 * y - 8) = 4 * y^3 - 6 * y^2 - 15 * y + 48 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l1778_177838


namespace NUMINAMATH_CALUDE_cubic_expression_value_l1778_177867

theorem cubic_expression_value (m : ℝ) (h : m^2 - m - 1 = 0) : 
  2*m^3 - 3*m^2 - m + 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l1778_177867


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1778_177850

/-- Given a triangle with two sides of lengths 3 and 4, and the third side being the root
    of x^2 - 12x + 35 = 0 that satisfies the triangle inequality, the perimeter is 12. -/
theorem triangle_perimeter : ∀ x : ℝ,
  x^2 - 12*x + 35 = 0 →
  x > 0 →
  x < 3 + 4 →
  x > |3 - 4| →
  3 + 4 + x = 12 := by
  sorry


end NUMINAMATH_CALUDE_triangle_perimeter_l1778_177850


namespace NUMINAMATH_CALUDE_edward_spent_sixteen_l1778_177801

/-- The amount of money Edward spent -/
def amount_spent (initial_amount remaining_amount : ℕ) : ℕ :=
  initial_amount - remaining_amount

/-- Theorem: Edward spent $16 -/
theorem edward_spent_sixteen :
  amount_spent 18 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_edward_spent_sixteen_l1778_177801


namespace NUMINAMATH_CALUDE_coins_left_l1778_177834

def pennies : ℕ := 42
def nickels : ℕ := 36
def dimes : ℕ := 15
def donated : ℕ := 66

theorem coins_left : pennies + nickels + dimes - donated = 27 := by
  sorry

end NUMINAMATH_CALUDE_coins_left_l1778_177834


namespace NUMINAMATH_CALUDE_gorillas_sent_to_different_zoo_l1778_177896

theorem gorillas_sent_to_different_zoo :
  let initial_animals : ℕ := 68
  let hippopotamus : ℕ := 1
  let rhinos : ℕ := 3
  let lion_cubs : ℕ := 8
  let meerkats : ℕ := 2 * lion_cubs
  let final_animals : ℕ := 90
  let gorillas_sent : ℕ := initial_animals + hippopotamus + rhinos + lion_cubs + meerkats - final_animals
  gorillas_sent = 6 :=
by sorry

end NUMINAMATH_CALUDE_gorillas_sent_to_different_zoo_l1778_177896


namespace NUMINAMATH_CALUDE_quadratic_form_constant_l1778_177888

theorem quadratic_form_constant (x : ℝ) :
  ∃ (a h k : ℝ), x^2 - 7*x = a*(x - h)^2 + k ∧ k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_constant_l1778_177888


namespace NUMINAMATH_CALUDE_alicia_sundae_cost_l1778_177856

/-- The cost of Alicia's peanut butter sundae given the prices of other sundaes and the final bill with tip -/
theorem alicia_sundae_cost (yvette_sundae brant_sundae josh_sundae : ℚ)
  (tip_percentage : ℚ) (final_bill : ℚ) :
  yvette_sundae = 9 →
  brant_sundae = 10 →
  josh_sundae = (17/2) →
  tip_percentage = (1/5) →
  final_bill = 42 →
  ∃ (alicia_sundae : ℚ),
    alicia_sundae = (final_bill / (1 + tip_percentage)) - (yvette_sundae + brant_sundae + josh_sundae) ∧
    alicia_sundae = (15/2) := by
  sorry

end NUMINAMATH_CALUDE_alicia_sundae_cost_l1778_177856


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1778_177893

-- Define the simple interest calculation function
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

-- State the theorem
theorem interest_rate_calculation (principal time interest : ℚ) 
  (h1 : principal = 8925)
  (h2 : time = 5)
  (h3 : interest = 4016.25)
  (h4 : simple_interest principal (9 : ℚ) time = interest) :
  ∃ (rate : ℚ), simple_interest principal rate time = interest ∧ rate = 9 := by
  sorry


end NUMINAMATH_CALUDE_interest_rate_calculation_l1778_177893


namespace NUMINAMATH_CALUDE_solve_system_l1778_177804

theorem solve_system (c d : ℤ) 
  (eq1 : 5 + c = 7 - d) 
  (eq2 : 6 + d = 10 + c) : 
  5 - c = 6 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l1778_177804


namespace NUMINAMATH_CALUDE_obtuse_triangles_in_17gon_l1778_177895

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A triangle formed by three vertices of a regular polygon -/
structure PolygonTriangle (n : ℕ) (polygon : RegularPolygon n) where
  v1 : Fin n
  v2 : Fin n
  v3 : Fin n

/-- Predicate to determine if a triangle is obtuse -/
def isObtuseTriangle (n : ℕ) (polygon : RegularPolygon n) (triangle : PolygonTriangle n polygon) : Prop :=
  sorry

/-- Count the number of obtuse triangles in a regular polygon -/
def countObtuseTriangles (n : ℕ) (polygon : RegularPolygon n) : ℕ :=
  sorry

theorem obtuse_triangles_in_17gon :
  ∀ (polygon : RegularPolygon 17),
  countObtuseTriangles 17 polygon = 476 :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangles_in_17gon_l1778_177895


namespace NUMINAMATH_CALUDE_complex_ratio_range_l1778_177863

theorem complex_ratio_range (x y : ℝ) :
  let z : ℂ := x + y * Complex.I
  let ratio := (z + 1) / (z + 2)
  (ratio.re / ratio.im = Real.sqrt 3) →
  (y / x ∈ Set.Icc ((Real.sqrt 3 * -3 - 4 * Real.sqrt 2) / 5) ((Real.sqrt 3 * -3 + 4 * Real.sqrt 2) / 5)) :=
by sorry

end NUMINAMATH_CALUDE_complex_ratio_range_l1778_177863


namespace NUMINAMATH_CALUDE_journey_remaining_distance_l1778_177836

/-- The remaining distance to be driven in a journey. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Proof that the remaining distance is 3610 miles. -/
theorem journey_remaining_distance :
  remaining_distance 9475 5865 = 3610 := by
  sorry

end NUMINAMATH_CALUDE_journey_remaining_distance_l1778_177836


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1778_177861

def A : Set ℤ := {-2, 0}
def B : Set ℤ := {-2, 3}

theorem union_of_A_and_B : A ∪ B = {-2, 0, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1778_177861


namespace NUMINAMATH_CALUDE_intersection_point_l1778_177827

/-- The line defined by the equation y = -7x + 9 -/
def line (x : ℝ) : ℝ := -7 * x + 9

/-- The y-axis is defined as the set of points with x-coordinate equal to 0 -/
def y_axis (p : ℝ × ℝ) : Prop := p.1 = 0

/-- A point is on the line if its y-coordinate equals the line function at its x-coordinate -/
def on_line (p : ℝ × ℝ) : Prop := p.2 = line p.1

theorem intersection_point :
  ∃! p : ℝ × ℝ, y_axis p ∧ on_line p ∧ p = (0, 9) := by sorry

end NUMINAMATH_CALUDE_intersection_point_l1778_177827


namespace NUMINAMATH_CALUDE_equation_solution_l1778_177899

theorem equation_solution (x : ℝ) : (x + 6) / (x - 3) = 4 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1778_177899


namespace NUMINAMATH_CALUDE_gina_hourly_rate_l1778_177814

/-- Gina's painting rates and order details -/
structure PaintingJob where
  rose_rate : ℕ  -- Cups with roses painted per hour
  lily_rate : ℕ  -- Cups with lilies painted per hour
  rose_order : ℕ  -- Number of rose cups ordered
  lily_order : ℕ  -- Number of lily cups ordered
  total_payment : ℕ  -- Total payment for the order in dollars

/-- Calculate Gina's hourly rate for a given painting job -/
def hourly_rate (job : PaintingJob) : ℚ :=
  job.total_payment / (job.rose_order / job.rose_rate + job.lily_order / job.lily_rate)

/-- Theorem: Gina's hourly rate for the given job is $30 -/
theorem gina_hourly_rate :
  let job : PaintingJob := {
    rose_rate := 6,
    lily_rate := 7,
    rose_order := 6,
    lily_order := 14,
    total_payment := 90
  }
  hourly_rate job = 30 := by
  sorry

end NUMINAMATH_CALUDE_gina_hourly_rate_l1778_177814


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l1778_177807

theorem sum_remainder_mod_seven :
  (9543 + 9544 + 9545 + 9546 + 9547) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l1778_177807


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_l1778_177874

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |2*x - 1|

-- Theorem 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Theorem 2
theorem range_of_a :
  ∀ a : ℝ, (∀ x ∈ {x : ℝ | 1/2 ≤ x ∧ x ≤ 1}, f a x ≤ |2*x + 1|) →
  -1 ≤ a ∧ a ≤ 5/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_l1778_177874


namespace NUMINAMATH_CALUDE_expression_evaluation_l1778_177864

theorem expression_evaluation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2) / (a * b) - (a^2 + a * b) / (a^2 + b^2) = 
  (a^4 + b^4 + a^2 * b^2 - a^2 * b - a * b^2) / (a * b * (a^2 + b^2)) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1778_177864


namespace NUMINAMATH_CALUDE_number_of_factors_of_N_l1778_177823

def N : ℕ := 17^3 + 3 * 17^2 + 3 * 17 + 1

theorem number_of_factors_of_N : (Finset.filter (· ∣ N) (Finset.range (N + 1))).card = 28 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_N_l1778_177823


namespace NUMINAMATH_CALUDE_percent_women_non_union_part_time_l1778_177898

/-- Represents the percentage of employees who are men -/
def percentMen : ℝ := 54

/-- Represents the percentage of employees who are women -/
def percentWomen : ℝ := 46

/-- Represents the percentage of men who work full-time -/
def percentMenFullTime : ℝ := 70

/-- Represents the percentage of men who work part-time -/
def percentMenPartTime : ℝ := 30

/-- Represents the percentage of women who work full-time -/
def percentWomenFullTime : ℝ := 60

/-- Represents the percentage of women who work part-time -/
def percentWomenPartTime : ℝ := 40

/-- Represents the percentage of full-time employees who are unionized -/
def percentFullTimeUnionized : ℝ := 60

/-- Represents the percentage of part-time employees who are unionized -/
def percentPartTimeUnionized : ℝ := 50

/-- The main theorem stating that given the conditions, 
    approximately 52.94% of non-union part-time employees are women -/
theorem percent_women_non_union_part_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧
  abs ((9 : ℝ) / 17 * 100 - 52.94) < ε := by
  sorry


end NUMINAMATH_CALUDE_percent_women_non_union_part_time_l1778_177898


namespace NUMINAMATH_CALUDE_trivia_team_absentees_l1778_177860

theorem trivia_team_absentees (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) : 
  total_members = 9 →
  points_per_member = 2 →
  total_points = 12 →
  total_members - (total_points / points_per_member) = 3 := by
sorry

end NUMINAMATH_CALUDE_trivia_team_absentees_l1778_177860


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1778_177839

-- Define the function f(x) = ax³ - 2x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * x

-- Theorem statement
theorem function_passes_through_point (a : ℝ) :
  f a (-1) = 4 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1778_177839


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l1778_177811

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l1778_177811


namespace NUMINAMATH_CALUDE_ice_skate_rental_fee_l1778_177837

/-- The rental fee for ice skates at a rink, given the admission fee, cost of new skates, and number of visits to justify buying. -/
theorem ice_skate_rental_fee 
  (admission_fee : ℚ) 
  (new_skates_cost : ℚ) 
  (visits_to_justify : ℕ) 
  (h1 : admission_fee = 5)
  (h2 : new_skates_cost = 65)
  (h3 : visits_to_justify = 26) :
  let rental_fee := (new_skates_cost + admission_fee * visits_to_justify) / visits_to_justify - admission_fee
  rental_fee = (5/2 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_ice_skate_rental_fee_l1778_177837


namespace NUMINAMATH_CALUDE_exam_average_l1778_177810

theorem exam_average (total_boys : ℕ) (passed_boys : ℕ) (passed_avg : ℚ) (failed_avg : ℚ) 
  (h1 : total_boys = 120)
  (h2 : passed_boys = 110)
  (h3 : passed_avg = 39)
  (h4 : failed_avg = 15) :
  (passed_avg * passed_boys + failed_avg * (total_boys - passed_boys)) / total_boys = 37 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_l1778_177810


namespace NUMINAMATH_CALUDE_max_temperature_range_l1778_177885

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem max_temperature_range 
  (T1 T2 T3 T4 T5 : ℕ) 
  (avg_temp : (T1 + T2 + T3 + T4 + T5) / 5 = 60)
  (lowest_temp : T1 = 50 ∧ T2 = 50)
  (consecutive : ∃ n : ℕ, T3 = n ∧ T4 = n + 1 ∧ T5 = n + 2)
  (ordered : T3 ≤ T4 ∧ T4 ≤ T5)
  (prime_exists : is_prime T3 ∨ is_prime T4 ∨ is_prime T5) :
  T5 - T1 = 18 :=
sorry

end NUMINAMATH_CALUDE_max_temperature_range_l1778_177885


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1778_177894

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The general form of a line equation: ax + by + c = 0 -/
structure GeneralLineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- Convert a line from slope-intercept form to general form -/
def toGeneralForm (l : Line) : GeneralLineEquation :=
  { a := l.slope, b := -1, c := l.y_intercept }

/-- The main theorem -/
theorem perpendicular_line_equation 
  (l : Line) 
  (h1 : l.y_intercept = 2) 
  (h2 : perpendicular l { slope := -1, y_intercept := 3 }) : 
  toGeneralForm l = { a := 1, b := -1, c := 2 } := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1778_177894


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l1778_177880

def number_of_lineups (total_players : ℕ) (captain_count : ℕ) (regular_players : ℕ) : ℕ :=
  total_players * (Nat.choose (total_players - 1) regular_players)

theorem basketball_lineup_count :
  number_of_lineups 12 1 5 = 5544 := by
sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l1778_177880


namespace NUMINAMATH_CALUDE_ball_ratio_l1778_177890

theorem ball_ratio (R B x : ℕ) : 
  R > 0 → B > 0 → x > 0 →
  R = (R + B + x) / 4 →
  R + x = (B + x) / 2 →
  R / B = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ball_ratio_l1778_177890


namespace NUMINAMATH_CALUDE_max_piles_for_660_stones_l1778_177873

/-- Represents the stone splitting game -/
structure StoneSplittingGame where
  initial_stones : ℕ
  max_piles : ℕ

/-- Checks if a list of pile sizes is valid according to the game rules -/
def is_valid_configuration (piles : List ℕ) : Prop :=
  ∀ i j, i < piles.length → j < piles.length → 
    2 * piles[i]! > piles[j]! ∧ 2 * piles[j]! > piles[i]!

/-- Theorem stating the maximum number of piles for 660 stones -/
theorem max_piles_for_660_stones (game : StoneSplittingGame) 
  (h1 : game.initial_stones = 660)
  (h2 : game.max_piles = 30) :
  ∃ (piles : List ℕ), 
    piles.length = game.max_piles ∧ 
    piles.sum = game.initial_stones ∧
    is_valid_configuration piles ∧
    ∀ (other_piles : List ℕ), 
      other_piles.sum = game.initial_stones → 
      is_valid_configuration other_piles →
      other_piles.length ≤ game.max_piles :=
sorry


end NUMINAMATH_CALUDE_max_piles_for_660_stones_l1778_177873


namespace NUMINAMATH_CALUDE_parabola_conditions_imply_a_range_l1778_177883

theorem parabola_conditions_imply_a_range (a : ℝ) : 
  (a - 1 > 0) →  -- parabola y=(a-1)x^2 opens upwards
  (2*a - 3 < 0) →  -- parabola y=(2a-3)x^2 opens downwards
  (|a - 1| > |2*a - 3|) →  -- parabola y=(a-1)x^2 has a wider opening than y=(2a-3)x^2
  (4/3 < a ∧ a < 3/2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_conditions_imply_a_range_l1778_177883


namespace NUMINAMATH_CALUDE_prime_factorization_sum_l1778_177808

theorem prime_factorization_sum (a b c d : ℕ) : 
  2^a * 3^b * 5^c * 11^d = 14850 → 3*a + 2*b + 4*c + 6*d = 23 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_sum_l1778_177808


namespace NUMINAMATH_CALUDE_seven_place_value_difference_l1778_177816

def number : ℕ := 54179759

def first_seven_place_value : ℕ := 10000
def second_seven_place_value : ℕ := 10

def first_seven_value : ℕ := 7 * first_seven_place_value
def second_seven_value : ℕ := 7 * second_seven_place_value

theorem seven_place_value_difference : 
  first_seven_value - second_seven_value = 69930 := by
  sorry

end NUMINAMATH_CALUDE_seven_place_value_difference_l1778_177816


namespace NUMINAMATH_CALUDE_log_quarter_of_sixteen_eq_neg_two_l1778_177825

-- Define the logarithm function for base 1/4
noncomputable def log_quarter (x : ℝ) : ℝ := Real.log x / Real.log (1/4)

-- State the theorem
theorem log_quarter_of_sixteen_eq_neg_two :
  log_quarter 16 = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_quarter_of_sixteen_eq_neg_two_l1778_177825


namespace NUMINAMATH_CALUDE_triangle_side_ratio_maximum_l1778_177815

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and the area of the triangle is (1/2)c^2, the maximum value of (a^2 + b^2 + c^2) / (ab) is 2√2. -/
theorem triangle_side_ratio_maximum (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (1/2) * c^2 = (1/2) * a * b * Real.sin C →
  (∃ (x : ℝ), (a^2 + b^2 + c^2) / (a * b) ≤ x) ∧
  (∀ (x : ℝ), (a^2 + b^2 + c^2) / (a * b) ≤ x → x ≥ 2 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_triangle_side_ratio_maximum_l1778_177815


namespace NUMINAMATH_CALUDE_sibling_height_l1778_177870

/-- Given information about Eliza and her siblings' heights, prove the height of the sibling with unknown height -/
theorem sibling_height (total_height : ℕ) (eliza_height : ℕ) (sibling1_height : ℕ) (sibling2_height : ℕ) :
  total_height = 330 ∧
  eliza_height = 68 ∧
  sibling1_height = 66 ∧
  sibling2_height = 66 ∧
  ∃ (unknown_sibling_height last_sibling_height : ℕ),
    unknown_sibling_height = eliza_height + 2 ∧
    total_height = eliza_height + sibling1_height + sibling2_height + unknown_sibling_height + last_sibling_height →
  ∃ (unknown_sibling_height : ℕ), unknown_sibling_height = 70 :=
by sorry

end NUMINAMATH_CALUDE_sibling_height_l1778_177870


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_given_l1778_177844

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - 3 * y + 5 = 0

-- Define the point that the new line passes through
def point : ℝ × ℝ := (-2, 1)

-- Define the new line
def new_line (x y : ℝ) : Prop := 2 * x - 3 * y + 7 = 0

-- Theorem statement
theorem line_through_point_parallel_to_given :
  (new_line point.1 point.2) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), new_line x₁ y₁ → new_line x₂ y₂ → 
    (x₂ - x₁) * 3 = (y₂ - y₁) * 2) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), given_line x₁ y₁ → given_line x₂ y₂ → 
    (x₂ - x₁) * 3 = (y₂ - y₁) * 2) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_given_l1778_177844


namespace NUMINAMATH_CALUDE_distance_after_three_minutes_l1778_177853

/-- The distance between two vehicles after a given time -/
def distance_between_vehicles (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v2 - v1) * t

theorem distance_after_three_minutes :
  let truck_speed : ℝ := 65
  let car_speed : ℝ := 85
  let time_in_hours : ℝ := 3 / 60
  distance_between_vehicles truck_speed car_speed time_in_hours = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_three_minutes_l1778_177853


namespace NUMINAMATH_CALUDE_homework_time_calculation_l1778_177842

/-- The time Max spent on biology homework in minutes -/
def biology_time : ℝ := 24

/-- The time Max spent on history homework in minutes -/
def history_time : ℝ := 1.5 * biology_time

/-- The time Max spent on chemistry homework in minutes -/
def chemistry_time : ℝ := biology_time * 0.7

/-- The time Max spent on English homework in minutes -/
def english_time : ℝ := 2 * (history_time + chemistry_time)

/-- The time Max spent on geography homework in minutes -/
def geography_time : ℝ := 3 * history_time + 0.75 * english_time

/-- The total time Max spent on homework in minutes -/
def total_homework_time : ℝ := biology_time + history_time + chemistry_time + english_time + geography_time

theorem homework_time_calculation :
  total_homework_time = 369.6 := by sorry

end NUMINAMATH_CALUDE_homework_time_calculation_l1778_177842


namespace NUMINAMATH_CALUDE_pi_half_irrational_l1778_177817

theorem pi_half_irrational : Irrational (Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_pi_half_irrational_l1778_177817


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1778_177828

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties :
  a ≠ 0 →
  (∀ x : ℝ, f a b c x ≠ x) →
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a > 0 → ∀ x : ℝ, f a b c (f a b c x) > x) ∧
  (a + b + c = 0 → ∀ x : ℝ, f a b c (f a b c x) < x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1778_177828


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l1778_177800

-- Define the line
def line (x : ℝ) : ℝ := -x + 1

-- Define the third quadrant
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Theorem statement
theorem line_not_in_third_quadrant :
  ∀ x : ℝ, ¬(third_quadrant x (line x)) :=
by
  sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l1778_177800


namespace NUMINAMATH_CALUDE_monopoly_produces_durable_iff_lowquality_cost_gt_six_l1778_177859

/-- Represents a coffee machine producer -/
structure Producer where
  isDurable : Bool
  cost : ℝ

/-- Represents a consumer of coffee machines -/
structure Consumer where
  benefit : ℝ
  periods : ℕ

/-- Represents the market for coffee machines -/
inductive Market
  | Monopoly
  | PerfectlyCompetitive

/-- Define the conditions for the coffee machine problem -/
def coffeeMachineProblem (c : Consumer) (pd : Producer) (pl : Producer) (m : Market) : Prop :=
  c.periods = 2 ∧ 
  c.benefit = 20 ∧
  pd.isDurable = true ∧
  pd.cost = 12 ∧
  pl.isDurable = false

/-- Theorem: A monopoly will produce only durable coffee machines if and only if 
    the average cost of producing a low-quality coffee machine is greater than 6 monetary units -/
theorem monopoly_produces_durable_iff_lowquality_cost_gt_six 
  (c : Consumer) (pd : Producer) (pl : Producer) (m : Market) :
  coffeeMachineProblem c pd pl Market.Monopoly →
  (∀ S, pl.cost = S → (pd.cost < pl.cost ↔ S > 6)) :=
sorry

end NUMINAMATH_CALUDE_monopoly_produces_durable_iff_lowquality_cost_gt_six_l1778_177859


namespace NUMINAMATH_CALUDE_resort_tips_l1778_177882

theorem resort_tips (total_months : ℕ) (other_months : ℕ) (avg_other_tips : ℝ) (aug_tips : ℝ) :
  total_months = other_months + 1 →
  aug_tips = 0.5 * (aug_tips + other_months * avg_other_tips) →
  aug_tips = 6 * avg_other_tips :=
by
  sorry

end NUMINAMATH_CALUDE_resort_tips_l1778_177882


namespace NUMINAMATH_CALUDE_inverse_of_A_l1778_177831

def A : Matrix (Fin 3) (Fin 3) ℚ := !![1, 2, 3; 0, -1, 2; 3, 0, 7]

def A_inverse : Matrix (Fin 3) (Fin 3) ℚ := !![-1/2, -1, 1/2; 3/7, -1/7, -1/7; 3/14, 3/7, -1/14]

theorem inverse_of_A : A⁻¹ = A_inverse := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l1778_177831


namespace NUMINAMATH_CALUDE_base_b_is_seven_l1778_177826

/-- Given that in base b, the square of 22_b is 514_b, prove that b = 7 -/
theorem base_b_is_seven (b : ℕ) (h : b > 1) : 
  (2 * b + 2)^2 = 5 * b^2 + b + 4 → b = 7 := by
  sorry

end NUMINAMATH_CALUDE_base_b_is_seven_l1778_177826


namespace NUMINAMATH_CALUDE_halloween_cleanup_time_halloween_cleanup_time_specific_l1778_177835

/-- Calculates the total cleaning time for Halloween vandalism -/
theorem halloween_cleanup_time 
  (egg_cleanup_time : ℕ) 
  (tp_cleanup_time : ℕ) 
  (num_eggs : ℕ) 
  (num_tp : ℕ) : ℕ :=
  let egg_time_seconds := egg_cleanup_time * num_eggs
  let egg_time_minutes := egg_time_seconds / 60
  let tp_time_minutes := tp_cleanup_time * num_tp
  egg_time_minutes + tp_time_minutes

/-- Proves that the total cleaning time for 60 eggs and 7 rolls of toilet paper is 225 minutes -/
theorem halloween_cleanup_time_specific : 
  halloween_cleanup_time 15 30 60 7 = 225 := by
  sorry

end NUMINAMATH_CALUDE_halloween_cleanup_time_halloween_cleanup_time_specific_l1778_177835


namespace NUMINAMATH_CALUDE_bicycle_time_calculation_l1778_177843

def total_distance : ℝ := 20
def bicycle_speed : ℝ := 30
def running_speed : ℝ := 8
def total_time : ℝ := 117

theorem bicycle_time_calculation (t : ℝ) 
  (h1 : t ≥ 0) 
  (h2 : t ≤ total_time) 
  (h3 : (t / 60) * bicycle_speed + ((total_time - t) / 60) * running_speed = total_distance) : 
  t = 12 := by sorry

end NUMINAMATH_CALUDE_bicycle_time_calculation_l1778_177843


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l1778_177887

theorem lcm_factor_problem (A B : ℕ+) (X : ℕ) : 
  Nat.gcd A B = 23 →
  A = 230 →
  Nat.lcm A B = 23 * X * 10 →
  X = 1 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l1778_177887


namespace NUMINAMATH_CALUDE_four_digit_count_l1778_177809

-- Define the range of four-digit numbers
def four_digit_start : ℕ := 1000
def four_digit_end : ℕ := 9999

-- Theorem statement
theorem four_digit_count : 
  (Finset.range (four_digit_end - four_digit_start + 1)).card = 9000 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_count_l1778_177809


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l1778_177897

def price_reduction_problem (original_price : ℝ) : Prop :=
  let reduced_price := 0.75 * original_price
  let original_amount := 1100 / original_price
  let new_amount := 1100 / reduced_price
  (new_amount - original_amount = 5) ∧ (reduced_price = 55)

theorem price_reduction_theorem :
  ∃ (original_price : ℝ), price_reduction_problem original_price :=
sorry

end NUMINAMATH_CALUDE_price_reduction_theorem_l1778_177897


namespace NUMINAMATH_CALUDE_group_average_score_l1778_177813

theorem group_average_score (class_average : ℝ) (differences : List ℝ) : 
  class_average = 80 →
  differences = [2, 3, -3, -5, 12, 14, 10, 4, -6, 4, -11, -7, 8, -2] →
  (class_average + (differences.sum / differences.length)) = 81.64 := by
sorry

end NUMINAMATH_CALUDE_group_average_score_l1778_177813


namespace NUMINAMATH_CALUDE_fourth_roll_eight_prob_l1778_177802

-- Define the probabilities for the fair die
def fair_die_prob : ℚ := 1 / 8

-- Define the probabilities for the biased die
def biased_die_prob_eight : ℚ := 3 / 4
def biased_die_prob_other : ℚ := 1 / 28

-- Define the probability of selecting each die
def die_selection_prob : ℚ := 1 / 2

-- Define the number of rolls
def num_rolls : ℕ := 4

-- Define the number of sides on each die
def num_sides : ℕ := 8

-- Define the theorem
theorem fourth_roll_eight_prob :
  let p_fair_three_eights : ℚ := fair_die_prob ^ 3
  let p_biased_three_eights : ℚ := biased_die_prob_eight ^ 3
  let p_three_eights : ℚ := die_selection_prob * p_fair_three_eights + die_selection_prob * p_biased_three_eights
  let p_fair_given_three_eights : ℚ := (die_selection_prob * p_fair_three_eights) / p_three_eights
  let p_biased_given_three_eights : ℚ := (die_selection_prob * p_biased_three_eights) / p_three_eights
  let p_fourth_eight : ℚ := p_fair_given_three_eights * fair_die_prob + p_biased_given_three_eights * biased_die_prob_eight
  p_fourth_eight = 1297 / 1736 :=
by sorry

end NUMINAMATH_CALUDE_fourth_roll_eight_prob_l1778_177802


namespace NUMINAMATH_CALUDE_ten_digit_numbers_with_repeats_l1778_177847

theorem ten_digit_numbers_with_repeats (n : ℕ) : n = 9 * 10^9 - 9 * Nat.factorial 9 :=
  by
    sorry

end NUMINAMATH_CALUDE_ten_digit_numbers_with_repeats_l1778_177847


namespace NUMINAMATH_CALUDE_sine_cosine_relation_l1778_177829

theorem sine_cosine_relation (x : ℝ) (h : Real.cos (5 * Real.pi / 6 - x) = 1 / 3) :
  Real.sin (x - Real.pi / 3) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_relation_l1778_177829


namespace NUMINAMATH_CALUDE_females_together_arrangements_l1778_177858

/-- Represents the number of students of each gender -/
def num_males : ℕ := 2
def num_females : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_males + num_females

/-- The number of ways to arrange the students with females next to each other -/
def arrangements_with_females_together : ℕ := 12

/-- Theorem stating that the number of arrangements with females together is 12 -/
theorem females_together_arrangements :
  (arrangements_with_females_together = 12) ∧
  (num_males = 2) ∧
  (num_females = 2) ∧
  (total_students = 4) := by
  sorry

end NUMINAMATH_CALUDE_females_together_arrangements_l1778_177858


namespace NUMINAMATH_CALUDE_angle_is_three_pi_over_four_l1778_177881

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_is_three_pi_over_four (a b : ℝ × ℝ) 
  (h1 : a.fst * (a.fst - 2 * b.fst) + a.snd * (a.snd - 2 * b.snd) = 3)
  (h2 : a.fst^2 + a.snd^2 = 1)
  (h3 : b = (1, 1)) :
  angle_between_vectors a b = 3 * π / 4 := by sorry

end NUMINAMATH_CALUDE_angle_is_three_pi_over_four_l1778_177881


namespace NUMINAMATH_CALUDE_factorize_difference_of_squares_l1778_177833

variable (R : Type*) [CommRing R]
variable (a x y : R)

theorem factorize_difference_of_squares : a * x^2 - a * y^2 = a * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorize_difference_of_squares_l1778_177833


namespace NUMINAMATH_CALUDE_invisible_dots_count_l1778_177822

/-- The sum of dots on a standard six-sided die -/
def standard_die_sum : Nat := 21

/-- The total number of dots on five standard six-sided dice -/
def total_dots (n : Nat) : Nat := n * standard_die_sum

/-- The sum of visible dots in the given configuration -/
def visible_dots : Nat := 1 + 2 + 3 + 4 + 5 + 6 + 2 + 3 + 4 + 5 + 6 + 4 + 5 + 6

/-- The number of dice in the problem -/
def num_dice : Nat := 5

/-- The number of visible faces in the problem -/
def num_visible_faces : Nat := 14

theorem invisible_dots_count :
  total_dots num_dice - visible_dots = 49 :=
sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l1778_177822


namespace NUMINAMATH_CALUDE_max_k_value_l1778_177846

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 5 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 22) / 2 := by
sorry

end NUMINAMATH_CALUDE_max_k_value_l1778_177846


namespace NUMINAMATH_CALUDE_certain_number_problem_l1778_177892

theorem certain_number_problem (x y : ℝ) (h1 : 0.25 * x = 0.15 * y - 15) (h2 : x = 840) : y = 1500 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1778_177892


namespace NUMINAMATH_CALUDE_derivative_of_product_l1778_177803

theorem derivative_of_product (x : ℝ) :
  deriv (fun x => (3 * x^2 - 4*x) * (2*x + 1)) x = 18 * x^2 - 10 * x - 4 := by
sorry

end NUMINAMATH_CALUDE_derivative_of_product_l1778_177803


namespace NUMINAMATH_CALUDE_cosine_in_right_triangle_l1778_177878

theorem cosine_in_right_triangle (D E F : Real) (h1 : 0 < D) (h2 : 0 < E) (h3 : 0 < F) : 
  D^2 + E^2 = F^2 → D = 8 → F = 17 → Real.cos (Real.arccos (D / F)) = 15 / 17 := by
sorry

end NUMINAMATH_CALUDE_cosine_in_right_triangle_l1778_177878


namespace NUMINAMATH_CALUDE_correct_arrangements_l1778_177849

/-- The number of ways to assign students to tasks with restrictions -/
def assignment_arrangements (n m : ℕ) (r : ℕ) : ℕ :=
  Nat.descFactorial n m - 2 * Nat.descFactorial (n - 1) (m - 1)

/-- Theorem stating the correct number of arrangements -/
theorem correct_arrangements :
  assignment_arrangements 6 4 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_correct_arrangements_l1778_177849


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1778_177845

theorem count_integers_satisfying_inequality :
  ∃! (S : Finset ℤ), (∀ n : ℤ, n ∈ S ↔ (Real.sqrt (n + 1) ≤ Real.sqrt (5*n - 7) ∧ Real.sqrt (5*n - 7) < Real.sqrt (3*n + 6))) ∧ S.card = 5 :=
sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1778_177845


namespace NUMINAMATH_CALUDE_function_nonpositive_implies_a_geq_3_l1778_177879

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 3 - a

-- State the theorem
theorem function_nonpositive_implies_a_geq_3 :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ 0) → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_function_nonpositive_implies_a_geq_3_l1778_177879


namespace NUMINAMATH_CALUDE_polygon_sides_l1778_177865

/-- Given a polygon with sum of interior angles equal to 1080°, prove it has 8 sides -/
theorem polygon_sides (sum_interior_angles : ℝ) (h : sum_interior_angles = 1080) : 
  (sum_interior_angles / 180 + 2 : ℝ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1778_177865


namespace NUMINAMATH_CALUDE_andrews_dog_foreign_objects_l1778_177875

/-- Calculates the total number of foreign objects on a dog given the number of burrs,
    the ratio of ticks to burrs, and the ratio of fleas to ticks. -/
def total_foreign_objects (burrs : ℕ) (ticks_to_burrs_ratio : ℕ) (fleas_to_ticks_ratio : ℕ) : ℕ :=
  let ticks := burrs * ticks_to_burrs_ratio
  let fleas := ticks * fleas_to_ticks_ratio
  burrs + ticks + fleas

/-- Theorem stating that for a dog with 12 burrs, 6 times as many ticks as burrs,
    and 3 times as many fleas as ticks, the total number of foreign objects is 300. -/
theorem andrews_dog_foreign_objects : 
  total_foreign_objects 12 6 3 = 300 := by
  sorry

end NUMINAMATH_CALUDE_andrews_dog_foreign_objects_l1778_177875


namespace NUMINAMATH_CALUDE_infinitely_many_even_floor_squares_l1778_177872

theorem infinitely_many_even_floor_squares (α : ℝ) (h : α > 0) :
  Set.Infinite {n : ℕ+ | Even ⌊(n : ℝ)^2 * α⌋} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_even_floor_squares_l1778_177872


namespace NUMINAMATH_CALUDE_inequality_satisfied_by_five_integers_l1778_177819

theorem inequality_satisfied_by_five_integers :
  ∃! (S : Finset ℤ), (∀ n ∈ S, Real.sqrt (n + 1 : ℝ) ≤ Real.sqrt (5 * n - 7 : ℝ) ∧
                               Real.sqrt (5 * n - 7 : ℝ) < Real.sqrt (3 * n + 6 : ℝ)) ∧
                     S.card = 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_satisfied_by_five_integers_l1778_177819


namespace NUMINAMATH_CALUDE_abc_sum_mod_five_l1778_177851

theorem abc_sum_mod_five (a b c : ℕ) : 
  a < 5 → b < 5 → c < 5 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 5 = 1 →
  (3 * c) % 5 = 1 →
  (4 * b) % 5 = (1 + b) % 5 →
  (a + b + c) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_mod_five_l1778_177851


namespace NUMINAMATH_CALUDE_arccos_sqrt2_over_2_l1778_177812

theorem arccos_sqrt2_over_2 : Real.arccos (Real.sqrt 2 / 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sqrt2_over_2_l1778_177812


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1778_177830

/-- Given a rectangular frame made with 240 cm of wire, where the ratio of
    length:width:height is 3:2:1, prove that the dimensions are 30 cm, 20 cm,
    and 10 cm respectively. -/
theorem rectangle_dimensions (total_wire : ℝ) (length width height : ℝ)
    (h1 : total_wire = 240)
    (h2 : length + width + height = total_wire / 4)
    (h3 : length = 3 * height)
    (h4 : width = 2 * height) :
    length = 30 ∧ width = 20 ∧ height = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1778_177830


namespace NUMINAMATH_CALUDE_driveways_shoveled_l1778_177832

-- Define the prices and quantities
def candy_bar_price : ℚ := 3/4
def candy_bar_quantity : ℕ := 2
def lollipop_price : ℚ := 1/4
def lollipop_quantity : ℕ := 4
def driveway_price : ℚ := 3/2

-- Define the total spent at the candy store
def total_spent : ℚ := candy_bar_price * candy_bar_quantity + lollipop_price * lollipop_quantity

-- Define the fraction of earnings spent
def fraction_spent : ℚ := 1/6

-- Theorem to prove
theorem driveways_shoveled :
  (total_spent / fraction_spent) / driveway_price = 10 := by
  sorry


end NUMINAMATH_CALUDE_driveways_shoveled_l1778_177832


namespace NUMINAMATH_CALUDE_number_problem_l1778_177886

theorem number_problem (x : ℝ) : (0.6 * (3/5) * x = 36) → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1778_177886


namespace NUMINAMATH_CALUDE_sphere_cube_volume_comparison_l1778_177862

theorem sphere_cube_volume_comparison :
  ∀ (r a : ℝ), r > 0 → a > 0 →
  4 * π * r^2 = 6 * a^2 →
  (4/3) * π * r^3 > a^3 := by
sorry

end NUMINAMATH_CALUDE_sphere_cube_volume_comparison_l1778_177862


namespace NUMINAMATH_CALUDE_broadcasting_methods_count_l1778_177821

/-- The number of different commercial advertisements -/
def num_commercial : ℕ := 3

/-- The number of different Olympic promotional advertisements -/
def num_olympic : ℕ := 2

/-- The total number of advertisements -/
def total_ads : ℕ := 5

/-- Function to calculate the number of broadcasting methods -/
def num_broadcasting_methods : ℕ :=
  -- The actual calculation is not implemented, as we only need the statement
  sorry

/-- Theorem stating that the number of broadcasting methods is 36 -/
theorem broadcasting_methods_count :
  num_broadcasting_methods = 36 :=
sorry

end NUMINAMATH_CALUDE_broadcasting_methods_count_l1778_177821


namespace NUMINAMATH_CALUDE_exist_four_distinct_numbers_perfect_squares_l1778_177871

theorem exist_four_distinct_numbers_perfect_squares : 
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ (m n : ℕ), 
      a^2 + 2*c*d + b^2 = m^2 ∧
      c^2 + 2*a*b + d^2 = n^2 :=
by sorry

end NUMINAMATH_CALUDE_exist_four_distinct_numbers_perfect_squares_l1778_177871


namespace NUMINAMATH_CALUDE_total_money_is_900_l1778_177876

/-- The amount of money Sam has -/
def sam_money : ℕ := 200

/-- The amount of money Billy has -/
def billy_money : ℕ := 3 * sam_money - 150

/-- The amount of money Lila has -/
def lila_money : ℕ := billy_money - sam_money

/-- The total amount of money they have together -/
def total_money : ℕ := sam_money + billy_money + lila_money

theorem total_money_is_900 : total_money = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_900_l1778_177876


namespace NUMINAMATH_CALUDE_antibiotics_cost_proof_l1778_177855

def antibiotics_problem (doses_per_day : ℕ) (days : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (doses_per_day * days)

theorem antibiotics_cost_proof (doses_per_day : ℕ) (days : ℕ) (total_cost : ℚ) 
  (h1 : doses_per_day = 3)
  (h2 : days = 7)
  (h3 : total_cost = 63) :
  antibiotics_problem doses_per_day days total_cost = 3 := by
sorry

end NUMINAMATH_CALUDE_antibiotics_cost_proof_l1778_177855


namespace NUMINAMATH_CALUDE_max_circle_sum_l1778_177824

/-- Represents the seven regions formed by the intersection of three circles -/
inductive Region
| A  -- shared by all three circles
| B  -- shared by two circles
| C  -- shared by two circles
| D  -- shared by two circles
| E  -- in one circle only
| F  -- in one circle only
| G  -- in one circle only

/-- Assignment of integers to regions -/
def Assignment := Region → Fin 7

/-- A circle is represented by the four regions it contains -/
structure Circle :=
  (r1 r2 r3 r4 : Region)

/-- The three circles in the problem -/
def circles : Fin 3 → Circle := sorry

/-- The sum of values in a circle for a given assignment -/
def circleSum (a : Assignment) (c : Circle) : ℕ :=
  a c.r1 + a c.r2 + a c.r3 + a c.r4

/-- An assignment is valid if all values are distinct -/
def validAssignment (a : Assignment) : Prop :=
  ∀ r1 r2 : Region, r1 ≠ r2 → a r1 ≠ a r2

/-- An assignment satisfies the equal sum condition -/
def satisfiesEqualSum (a : Assignment) : Prop :=
  ∀ c1 c2 : Fin 3, circleSum a (circles c1) = circleSum a (circles c2)

/-- The maximum possible sum for each circle -/
def maxSum : ℕ := 15

theorem max_circle_sum :
  ∃ (a : Assignment), validAssignment a ∧ satisfiesEqualSum a ∧
  (∀ c : Fin 3, circleSum a (circles c) = maxSum) ∧
  (∀ (a' : Assignment), validAssignment a' ∧ satisfiesEqualSum a' →
    ∀ c : Fin 3, circleSum a' (circles c) ≤ maxSum) := by
  sorry

end NUMINAMATH_CALUDE_max_circle_sum_l1778_177824


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l1778_177854

theorem smallest_number_satisfying_conditions : ∃ n : ℕ, n > 0 ∧ 
  n % 6 = 2 ∧ n % 8 = 3 ∧ n % 9 = 4 ∧ 
  ∀ m : ℕ, m > 0 → m % 6 = 2 → m % 8 = 3 → m % 9 = 4 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l1778_177854


namespace NUMINAMATH_CALUDE_unique_prime_sum_of_fourth_powers_l1778_177889

theorem unique_prime_sum_of_fourth_powers (p a b c : ℕ) : 
  Prime p ∧ Prime a ∧ Prime b ∧ Prime c ∧ p = a^4 + b^4 + c^4 - 3 → p = 719 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_sum_of_fourth_powers_l1778_177889


namespace NUMINAMATH_CALUDE_rice_bags_sold_l1778_177869

/-- A trader sells rice bags and restocks. This theorem proves the number of bags sold. -/
theorem rice_bags_sold (initial_stock : ℕ) (restocked : ℕ) (final_stock : ℕ) 
  (h1 : initial_stock = 55)
  (h2 : restocked = 132)
  (h3 : final_stock = 164) :
  initial_stock + restocked - final_stock = 23 := by
  sorry

end NUMINAMATH_CALUDE_rice_bags_sold_l1778_177869


namespace NUMINAMATH_CALUDE_quadratic_range_l1778_177877

/-- The quadratic function f(x) = x^2 - 2x - 3 -/
def f (x : ℝ) := x^2 - 2*x - 3

/-- The theorem states that for x in [-2, 2], the range of f(x) is [-4, 5] -/
theorem quadratic_range :
  ∀ x ∈ Set.Icc (-2 : ℝ) 2,
  ∃ y ∈ Set.Icc (-4 : ℝ) 5,
  f x = y ∧
  (∀ z, f z ∈ Set.Icc (-4 : ℝ) 5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_range_l1778_177877


namespace NUMINAMATH_CALUDE_polygon_equal_sides_different_angles_l1778_177841

-- Define a polygon type
inductive Polygon
| Triangle
| Quadrilateral
| Pentagon

-- Function to check if a polygon can have all sides equal and all angles different
def canHaveEqualSidesAndDifferentAngles (p : Polygon) : Prop :=
  match p with
  | Polygon.Triangle => False
  | Polygon.Quadrilateral => False
  | Polygon.Pentagon => True

-- Theorem statement
theorem polygon_equal_sides_different_angles :
  (canHaveEqualSidesAndDifferentAngles Polygon.Triangle = False) ∧
  (canHaveEqualSidesAndDifferentAngles Polygon.Quadrilateral = False) ∧
  (canHaveEqualSidesAndDifferentAngles Polygon.Pentagon = True) := by
  sorry

#check polygon_equal_sides_different_angles

end NUMINAMATH_CALUDE_polygon_equal_sides_different_angles_l1778_177841


namespace NUMINAMATH_CALUDE_factorial_division_l1778_177848

theorem factorial_division :
  (10 : ℕ).factorial / (5 : ℕ).factorial = 30240 :=
by
  -- Given: 10! = 3628800
  have h1 : (10 : ℕ).factorial = 3628800 := by sorry
  
  -- Definition of 5!
  have h2 : (5 : ℕ).factorial = 120 := by sorry
  
  -- Proof that 10! / 5! = 30240
  sorry

end NUMINAMATH_CALUDE_factorial_division_l1778_177848


namespace NUMINAMATH_CALUDE_topsoil_cost_theorem_l1778_177820

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The amount of topsoil in cubic yards -/
def amount_in_cubic_yards : ℝ := 7

/-- The cost of topsoil for a given amount of cubic yards -/
def topsoil_cost (amount : ℝ) : ℝ :=
  amount * cubic_yards_to_cubic_feet * cost_per_cubic_foot

theorem topsoil_cost_theorem :
  topsoil_cost amount_in_cubic_yards = 1512 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_theorem_l1778_177820


namespace NUMINAMATH_CALUDE_max_distance_ellipse_circle_l1778_177868

/-- The maximum distance between points on an ellipse and a moving circle --/
theorem max_distance_ellipse_circle (a b R : ℝ) (ha : 0 < b) (hab : b < a) (hR : b < R) (hRa : R < a) :
  let ellipse := {p : ℝ × ℝ | (p.1 / a)^2 + (p.2 / b)^2 = 1}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = R^2}
  ∃ (A B : ℝ × ℝ), A ∈ ellipse ∧ B ∈ circle ∧
    (∀ (C : ℝ × ℝ), C ∈ ellipse → (A.1 - C.1) * (B.2 - A.2) = (A.2 - C.2) * (B.1 - A.1)) ∧
    (∀ (D : ℝ × ℝ), D ∈ circle → (B.1 - D.1) * (A.2 - B.2) = (B.2 - D.2) * (A.1 - B.1)) ∧
    ∀ (A' B' : ℝ × ℝ), A' ∈ ellipse → B' ∈ circle →
      (∀ (C : ℝ × ℝ), C ∈ ellipse → (A'.1 - C.1) * (B'.2 - A'.2) = (A'.2 - C.2) * (B'.1 - A'.1)) →
      (∀ (D : ℝ × ℝ), D ∈ circle → (B'.1 - D.1) * (A'.2 - B'.2) = (B'.2 - D.2) * (A'.1 - B'.1)) →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) :=
by
  sorry

#check max_distance_ellipse_circle

end NUMINAMATH_CALUDE_max_distance_ellipse_circle_l1778_177868


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1778_177891

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (10 * x₁^2 - 14 * x₁ - 24 = 0) → 
  (10 * x₂^2 - 14 * x₂ - 24 = 0) → 
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 169/25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1778_177891


namespace NUMINAMATH_CALUDE_eleventh_term_is_768_l1778_177806

/-- A geometric sequence with given conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, ∃ q : ℝ, a (n + 1) = q * a n
  a_4 : a 4 = 6
  a_7 : a 7 = 48

/-- The 11th term of the geometric sequence is 768 -/
theorem eleventh_term_is_768 (seq : GeometricSequence) : seq.a 11 = 768 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_is_768_l1778_177806


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l1778_177840

/-- Represents a systematic sample from a range of products. -/
structure SystematicSample where
  total_products : Nat
  sample_size : Nat
  start : Nat
  step : Nat

/-- Generates a systematic sample. -/
def generateSample (s : SystematicSample) : List Nat :=
  List.range s.sample_size |>.map (fun i => s.start + i * s.step)

/-- Checks if a sample is valid for the given total number of products. -/
def isValidSample (sample : List Nat) (total_products : Nat) : Prop :=
  sample.all (· < total_products) ∧ sample.length > 0 ∧ sample.Nodup

/-- Theorem: The correct systematic sample for 50 products with 5 samples is [1, 11, 21, 31, 41]. -/
theorem correct_systematic_sample :
  let sample := [1, 11, 21, 31, 41]
  let s : SystematicSample := {
    total_products := 50,
    sample_size := 5,
    start := 1,
    step := 10
  }
  generateSample s = sample ∧ isValidSample sample s.total_products := by
  sorry


end NUMINAMATH_CALUDE_correct_systematic_sample_l1778_177840


namespace NUMINAMATH_CALUDE_circle_construction_l1778_177852

/-- Given a circle k0 with diameter AB and center O0, and additional circles k1, k2, k3, k4, k5, k6
    constructed as described in the problem, prove that their radii are in specific ratios to r0. -/
theorem circle_construction (r0 : ℝ) (r1 r2 r3 r4 r5 r6 : ℝ) 
  (h1 : r0 > 0)
  (h2 : r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ r4 > 0 ∧ r5 > 0 ∧ r6 > 0)
  (h3 : ∃ (A B O0 : ℝ × ℝ), ‖A - B‖ = 2 * r0 ∧ O0 = (A + B) / 2)
  (h4 : ∃ (k1 k1' : Set (ℝ × ℝ)), k1 ∩ k1' = {O0}) :
  r1 = r0 / 2 ∧ r2 = r0 / 3 ∧ r3 = r0 / 6 ∧ r4 = r0 / 4 ∧ r5 = r0 / 7 ∧ r6 = r0 / 8 := by
  sorry


end NUMINAMATH_CALUDE_circle_construction_l1778_177852


namespace NUMINAMATH_CALUDE_cycling_distance_l1778_177805

/-- Proves that cycling at a constant rate of 4 miles per hour for 2 hours results in a total distance of 8 miles. -/
theorem cycling_distance (rate : ℝ) (time : ℝ) (distance : ℝ) : 
  rate = 4 → time = 2 → distance = rate * time → distance = 8 := by
  sorry

#check cycling_distance

end NUMINAMATH_CALUDE_cycling_distance_l1778_177805


namespace NUMINAMATH_CALUDE_unique_solution_l1778_177857

/-- The vector [2, -3] -/
def v : Fin 2 → ℝ := ![2, -3]

/-- The vector [4, 7] -/
def w : Fin 2 → ℝ := ![4, 7]

/-- The equation to be solved -/
def equation (k : ℝ) : Prop :=
  ‖k • v - w‖ = 2 * Real.sqrt 13

/-- Theorem stating that k = -1 is the only solution -/
theorem unique_solution :
  ∃! k : ℝ, equation k ∧ k = -1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1778_177857


namespace NUMINAMATH_CALUDE_no_odd_faced_odd_edged_polyhedron_l1778_177818

/-- Represents a face of a polyhedron -/
structure Face where
  edges : Nat
  odd_edges : Odd edges

/-- Represents a polyhedron -/
structure Polyhedron where
  faces : List Face
  odd_faces : Odd faces.length

/-- Theorem stating that a polyhedron with an odd number of faces, 
    each having an odd number of edges, cannot exist -/
theorem no_odd_faced_odd_edged_polyhedron : 
  ¬ ∃ (p : Polyhedron), True := by sorry

end NUMINAMATH_CALUDE_no_odd_faced_odd_edged_polyhedron_l1778_177818
