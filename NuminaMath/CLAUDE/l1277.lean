import Mathlib

namespace NUMINAMATH_CALUDE_train_length_calculation_l1277_127713

/-- Two trains of equal length passing each other -/
def train_passing_problem (faster_speed slower_speed : ℝ) (passing_time : ℝ) : Prop :=
  faster_speed > slower_speed ∧
  faster_speed = 75 ∧
  slower_speed = 60 ∧
  passing_time = 45 ∧
  ∃ (train_length : ℝ),
    train_length = (faster_speed - slower_speed) * passing_time * (5 / 18) / 2

theorem train_length_calculation (faster_speed slower_speed passing_time : ℝ) :
  train_passing_problem faster_speed slower_speed passing_time →
  ∃ (train_length : ℝ), train_length = 93.75 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l1277_127713


namespace NUMINAMATH_CALUDE_distance_to_parabola_directrix_l1277_127707

/-- The distance from a point to the directrix of a parabola -/
def distance_to_directrix (a : ℝ) (P : ℝ × ℝ) : ℝ :=
  |P.1 + a|

/-- The parabola equation -/
def is_parabola (x y : ℝ) (a : ℝ) : Prop :=
  y^2 = -4*a*x

theorem distance_to_parabola_directrix :
  ∃ (a : ℝ), 
    is_parabola (-2) 4 a ∧ 
    distance_to_directrix a (-2, 4) = 4 :=
sorry

end NUMINAMATH_CALUDE_distance_to_parabola_directrix_l1277_127707


namespace NUMINAMATH_CALUDE_at_least_one_positive_l1277_127726

theorem at_least_one_positive (x y z : ℝ) : 
  let a := x^2 - 2*y + π/2
  let b := y^2 - 2*z + π/4
  let c := z^2 - 2*x + π/4
  max a (max b c) > 0 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_positive_l1277_127726


namespace NUMINAMATH_CALUDE_intersection_k_value_l1277_127715

theorem intersection_k_value (k : ℝ) : 
  (∃ y : ℝ, 3 * (-6) - 2 * y = k ∧ -6 - 0.5 * y = 10) → k = 46 := by
  sorry

end NUMINAMATH_CALUDE_intersection_k_value_l1277_127715


namespace NUMINAMATH_CALUDE_ellipse_dot_product_bounds_l1277_127743

/-- Given an ellipse with specific properties, prove that the dot product of vectors AP and FP is bounded. -/
theorem ellipse_dot_product_bounds (a b : ℝ) (h_ab : a > b) (h_b_pos : b > 0)
  (h_top_focus : 2 = Real.sqrt (a^2 - b^2))
  (h_eccentricity : (Real.sqrt (a^2 - b^2)) / a = 1/2) :
  ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 →
  0 ≤ (x + 2) * (x + 1) + y^2 ∧ (x + 2) * (x + 1) + y^2 ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_bounds_l1277_127743


namespace NUMINAMATH_CALUDE_game_result_l1277_127788

def f (n : ℕ) : ℕ :=
  if n ^ 2 = n then 8
  else if n % 3 = 0 then 4
  else if n % 2 = 0 then 1
  else 0

def allie_rolls : List ℕ := [3, 4, 6, 1]
def betty_rolls : List ℕ := [4, 2, 5, 1]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem game_result : total_points allie_rolls * total_points betty_rolls = 117 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l1277_127788


namespace NUMINAMATH_CALUDE_hyperbola_a_value_l1277_127777

/-- A hyperbola with equation x²/(a-3) + y²/(2-a) = 1, foci on the y-axis, and focal distance 4 -/
structure Hyperbola where
  a : ℝ
  equation : ∀ x y : ℝ, x^2 / (a - 3) + y^2 / (2 - a) = 1
  foci_on_y_axis : True  -- This is a placeholder for the foci condition
  focal_distance : ℝ
  focal_distance_value : focal_distance = 4

/-- The value of 'a' for the given hyperbola is 1/2 -/
theorem hyperbola_a_value (h : Hyperbola) : h.a = 1/2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_a_value_l1277_127777


namespace NUMINAMATH_CALUDE_people_arrangement_l1277_127721

/-- Given a total of 1600 people and columns of 85 people each, prove:
    1. The number of complete columns
    2. The number of people in the incomplete column
    3. The total number of rows
    4. The row in which the last person stands -/
theorem people_arrangement (total_people : ℕ) (people_per_column : ℕ) 
    (h1 : total_people = 1600)
    (h2 : people_per_column = 85) :
    let complete_columns := total_people / people_per_column
    let remaining_people := total_people % people_per_column
    (complete_columns = 18) ∧ 
    (remaining_people = 70) ∧
    (remaining_people = 70) ∧
    (remaining_people = 70) := by
  sorry

end NUMINAMATH_CALUDE_people_arrangement_l1277_127721


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l1277_127729

theorem perfect_square_polynomial (m : ℤ) : 
  1 + 2*m + 3*m^2 + 4*m^3 + 5*m^4 + 4*m^5 + 3*m^6 + 2*m^7 + m^8 = (1 + m + m^2 + m^3 + m^4)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l1277_127729


namespace NUMINAMATH_CALUDE_min_score_to_tie_record_l1277_127776

/-- Proves that the minimum average score per player in the final round to tie the league record
    is 12.5833 points less than the current league record. -/
theorem min_score_to_tie_record (
  league_record : ℝ)
  (team_size : ℕ)
  (season_length : ℕ)
  (current_score : ℝ)
  (bonus_points : ℕ)
  (h1 : league_record = 287.5)
  (h2 : team_size = 6)
  (h3 : season_length = 12)
  (h4 : current_score = 19350.5)
  (h5 : bonus_points = 300)
  : ∃ (min_score : ℝ), 
    league_record - min_score = 12.5833 ∧ 
    min_score * team_size + current_score + bonus_points = league_record * (team_size * season_length) :=
by
  sorry

end NUMINAMATH_CALUDE_min_score_to_tie_record_l1277_127776


namespace NUMINAMATH_CALUDE_line_slope_is_one_l1277_127780

theorem line_slope_is_one : 
  let line_eq := fun (x y : ℝ) => x - y + 1 = 0
  ∃ m : ℝ, (∀ x₁ y₁ x₂ y₂ : ℝ, 
    line_eq x₁ y₁ ∧ line_eq x₂ y₂ ∧ x₁ ≠ x₂ → 
    m = (y₂ - y₁) / (x₂ - x₁)) ∧ m = 1 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_is_one_l1277_127780


namespace NUMINAMATH_CALUDE_only_setB_forms_triangle_l1277_127744

/-- Represents a set of three line segments --/
structure LineSegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a set of line segments can form a triangle --/
def canFormTriangle (s : LineSegmentSet) : Prop :=
  s.a + s.b > s.c ∧ s.b + s.c > s.a ∧ s.c + s.a > s.b

/-- The given sets of line segments --/
def setA : LineSegmentSet := ⟨1, 2, 4⟩
def setB : LineSegmentSet := ⟨4, 6, 8⟩
def setC : LineSegmentSet := ⟨5, 6, 12⟩
def setD : LineSegmentSet := ⟨2, 3, 5⟩

/-- Theorem: Only set B can form a triangle --/
theorem only_setB_forms_triangle :
  ¬(canFormTriangle setA) ∧
  canFormTriangle setB ∧
  ¬(canFormTriangle setC) ∧
  ¬(canFormTriangle setD) :=
by sorry

end NUMINAMATH_CALUDE_only_setB_forms_triangle_l1277_127744


namespace NUMINAMATH_CALUDE_root_power_sums_equal_l1277_127761

-- Define the polynomial
def p (x : ℂ) : ℂ := x^3 + 2*x^2 + 3*x + 4

-- Define the sum of nth powers of roots
def S (n : ℕ) : ℂ := sorry

theorem root_power_sums_equal :
  S 1 = -2 ∧ S 2 = -2 ∧ S 3 = -2 := by sorry

end NUMINAMATH_CALUDE_root_power_sums_equal_l1277_127761


namespace NUMINAMATH_CALUDE_ant_meeting_point_l1277_127772

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point on the perimeter of the triangle -/
structure PerimeterPoint where
  distanceFromP : ℝ

/-- The theorem statement -/
theorem ant_meeting_point (t : Triangle) (s : PerimeterPoint) : 
  t.a = 7 ∧ t.b = 8 ∧ t.c = 9 →
  s.distanceFromP = (t.a + t.b + t.c) / 2 →
  s.distanceFromP - t.a = 5 := by
  sorry

end NUMINAMATH_CALUDE_ant_meeting_point_l1277_127772


namespace NUMINAMATH_CALUDE_numbers_close_to_zero_not_set_l1277_127718

-- Define the criteria for a set
structure SetCriteria where
  definiteness : Bool
  distinctness : Bool
  unorderedness : Bool

-- Define a predicate for whether a collection can form a set
def canFormSet (c : SetCriteria) : Bool :=
  c.definiteness ∧ c.distinctness ∧ c.unorderedness

-- Define the property of being "close to 0"
def closeToZero (ε : ℝ) (x : ℝ) : Prop := abs x < ε

-- Theorem stating that "Numbers close to 0" cannot form a set
theorem numbers_close_to_zero_not_set : 
  ∃ ε > 0, ¬∃ (S : Set ℝ), (∀ x ∈ S, closeToZero ε x) ∧ 
  (canFormSet ⟨true, true, true⟩) :=
sorry

end NUMINAMATH_CALUDE_numbers_close_to_zero_not_set_l1277_127718


namespace NUMINAMATH_CALUDE_congruence_and_infinite_primes_l1277_127779

theorem congruence_and_infinite_primes (p : ℕ) (hp : Prime p) (hp3 : p > 3) :
  (∃ x : ℕ, (x^2 + x + 1) % p = 0) →
  (p % 6 = 1 ∧ ∀ n : ℕ, ∃ q > n, Prime q ∧ q % 6 = 1) := by
  sorry

end NUMINAMATH_CALUDE_congruence_and_infinite_primes_l1277_127779


namespace NUMINAMATH_CALUDE_smallest_mu_inequality_l1277_127750

theorem smallest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  ∃ μ : ℝ, (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a^2 + b^2 + c^2 + d^2 ≥ 2*a*b + μ*b*c + 2*c*d) ∧
  (∀ μ' : ℝ, (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a^2 + b^2 + c^2 + d^2 ≥ 2*a*b + μ'*b*c + 2*c*d) → μ' ≥ μ) ∧
  μ = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_mu_inequality_l1277_127750


namespace NUMINAMATH_CALUDE_system_of_equations_range_l1277_127705

theorem system_of_equations_range (x y m : ℝ) : 
  x + 2*y = 1 - m →
  2*x + y = 3 →
  x + y > 0 →
  m < 4 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_range_l1277_127705


namespace NUMINAMATH_CALUDE_square_to_rectangle_area_ratio_l1277_127762

/-- A rectangle with a square inside it -/
structure RectangleWithSquare where
  square_side : ℝ
  rect_width : ℝ
  rect_length : ℝ
  width_to_side_ratio : rect_width = 3 * square_side
  length_to_width_ratio : rect_length = 2 * rect_width

/-- The theorem stating that the area of the square is 1/18 of the area of the rectangle -/
theorem square_to_rectangle_area_ratio (r : RectangleWithSquare) :
  (r.square_side ^ 2) / (r.rect_width * r.rect_length) = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_square_to_rectangle_area_ratio_l1277_127762


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1277_127730

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (a5_eq_6 : a 5 = 6)
  (a3_eq_2 : a 3 = 2) :
  ∀ n, a (n + 1) - a n = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1277_127730


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l1277_127733

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l1277_127733


namespace NUMINAMATH_CALUDE_time_to_fill_leaking_basin_l1277_127766

/-- Calculates the time to fill a leaking basin from a waterfall -/
theorem time_to_fill_leaking_basin 
  (waterfall_flow : ℝ) 
  (basin_capacity : ℝ) 
  (leak_rate : ℝ) 
  (h1 : waterfall_flow = 24)
  (h2 : basin_capacity = 260)
  (h3 : leak_rate = 4) : 
  basin_capacity / (waterfall_flow - leak_rate) = 13 := by
sorry


end NUMINAMATH_CALUDE_time_to_fill_leaking_basin_l1277_127766


namespace NUMINAMATH_CALUDE_class_size_theorem_l1277_127720

theorem class_size_theorem :
  ∀ (m d : ℕ),
  (∃ (r : ℕ), r = 3 * m ∧ r = 5 * d) →
  30 < m + d →
  m + d < 40 →
  m + d = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_theorem_l1277_127720


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1277_127773

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x^2 - x - 2 = 0 → -1 ≤ x ∧ x ≤ 2) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ x^2 - x - 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1277_127773


namespace NUMINAMATH_CALUDE_ratio_problem_l1277_127752

theorem ratio_problem (a b c : ℝ) 
  (hab : a / b = 11 / 3) 
  (hac : a / c = 11 / 15) : 
  b / c = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l1277_127752


namespace NUMINAMATH_CALUDE_smaller_square_area_half_larger_l1277_127747

/-- A circle with an inscribed square and a smaller square -/
structure SquaresInCircle where
  /-- The radius of the circle -/
  R : ℝ
  /-- The side length of the larger square -/
  a : ℝ
  /-- The side length of the smaller square -/
  b : ℝ
  /-- The larger square is inscribed in the circle -/
  h1 : R = a * Real.sqrt 2 / 2
  /-- The smaller square has one side coinciding with a side of the larger square -/
  h2 : b ≤ a
  /-- The smaller square has two vertices on the circle -/
  h3 : R^2 = (a/2 - b/2)^2 + b^2
  /-- The side length of the larger square is 4 units -/
  h4 : a = 4

/-- The area of the smaller square is half the area of the larger square -/
theorem smaller_square_area_half_larger (sq : SquaresInCircle) : 
  sq.b^2 = sq.a^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_square_area_half_larger_l1277_127747


namespace NUMINAMATH_CALUDE_max_a_value_l1277_127728

theorem max_a_value (a b : ℕ) (ha : 1 < a) (hb : a < b) :
  (∃! x : ℝ, -2 * x + 4033 = |x - 1| + |x + a| + |x - b|) →
  (∀ c : ℕ, (1 < c ∧ c < b ∧
    (∃! x : ℝ, -2 * x + 4033 = |x - 1| + |x + c| + |x - b|)) →
    c ≤ a) →
  a = 4031 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l1277_127728


namespace NUMINAMATH_CALUDE_number_count_with_incorrect_average_l1277_127792

theorem number_count_with_incorrect_average (n : ℕ) : 
  (n : ℝ) * 40.2 - (n : ℝ) * 40.1 = 35 → n = 350 := by
  sorry

end NUMINAMATH_CALUDE_number_count_with_incorrect_average_l1277_127792


namespace NUMINAMATH_CALUDE_croissant_making_time_l1277_127725

/-- Proves that the total time for making croissants is 6 hours -/
theorem croissant_making_time : 
  let fold_time : ℕ := 4 * 5
  let rest_time : ℕ := 4 * 75
  let mix_time : ℕ := 10
  let bake_time : ℕ := 30
  let minutes_per_hour : ℕ := 60
  (fold_time + rest_time + mix_time + bake_time) / minutes_per_hour = 6 := by
  sorry

end NUMINAMATH_CALUDE_croissant_making_time_l1277_127725


namespace NUMINAMATH_CALUDE_expression_evaluation_l1277_127711

theorem expression_evaluation :
  Real.sqrt 8 + (1/2)⁻¹ - 2 * Real.sin (45 * π / 180) - abs (1 - Real.sqrt 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1277_127711


namespace NUMINAMATH_CALUDE_new_selling_price_l1277_127790

theorem new_selling_price (old_price : ℝ) (old_profit_rate new_profit_rate : ℝ) : 
  old_price = 88 →
  old_profit_rate = 0.1 →
  new_profit_rate = 0.15 →
  let cost := old_price / (1 + old_profit_rate)
  let new_price := cost * (1 + new_profit_rate)
  new_price = 92 := by
sorry

end NUMINAMATH_CALUDE_new_selling_price_l1277_127790


namespace NUMINAMATH_CALUDE_total_profit_calculation_l1277_127709

/-- Given the capital ratios and R's share of the profit, calculate the total profit -/
theorem total_profit_calculation (P Q R : ℕ) (r_profit : ℕ) 
  (h1 : 4 * P = 6 * Q)
  (h2 : 6 * Q = 10 * R)
  (h3 : r_profit = 900) : 
  4650 = (31 * r_profit) / 6 :=
by sorry

end NUMINAMATH_CALUDE_total_profit_calculation_l1277_127709


namespace NUMINAMATH_CALUDE_annual_salary_calculation_l1277_127723

theorem annual_salary_calculation (hourly_wage : ℝ) (hours_per_day : ℕ) (days_per_month : ℕ) :
  hourly_wage = 8.50 →
  hours_per_day = 8 →
  days_per_month = 20 →
  hourly_wage * hours_per_day * days_per_month * 12 = 16320 :=
by
  sorry

end NUMINAMATH_CALUDE_annual_salary_calculation_l1277_127723


namespace NUMINAMATH_CALUDE_arrangement_count_proof_l1277_127735

/-- The number of ways to arrange 4 distinct digits in a 2 × 3 grid with 2 empty cells -/
def arrangement_count : ℕ := 360

/-- The size of the grid -/
def grid_size : ℕ × ℕ := (2, 3)

/-- The number of available digits -/
def digit_count : ℕ := 4

/-- The number of empty cells -/
def empty_cell_count : ℕ := 2

/-- The total number of cells in the grid -/
def total_cells : ℕ := grid_size.1 * grid_size.2

theorem arrangement_count_proof :
  arrangement_count = (Nat.choose total_cells empty_cell_count) * (Nat.factorial digit_count) :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_proof_l1277_127735


namespace NUMINAMATH_CALUDE_tangent_line_slope_l1277_127708

/-- The value of a for which the tangent line to y = ax - ln(x+1) at (0,0) is y = 2x -/
theorem tangent_line_slope (a : ℝ) : 
  (∀ x y : ℝ, y = a * x - Real.log (x + 1)) →
  (∃ m : ℝ, ∀ x y : ℝ, y = m * x ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → 
      |((a * h - Real.log (h + 1)) / h) - m| < ε)) →
  m = 2 →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l1277_127708


namespace NUMINAMATH_CALUDE_blue_marbles_most_numerous_l1277_127740

/-- Given a set of marbles with specific conditions, prove that blue marbles are the most numerous -/
theorem blue_marbles_most_numerous (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) 
  (h_total : total = 24)
  (h_red : red = total / 4)
  (h_blue : blue = red + 6)
  (h_yellow : yellow = total - red - blue) :
  blue > red ∧ blue > yellow := by
  sorry

end NUMINAMATH_CALUDE_blue_marbles_most_numerous_l1277_127740


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l1277_127769

theorem unique_solution_quadratic_inequality (p : ℝ) : 
  (∃! x : ℝ, 0 ≤ x^2 + p*x + 5 ∧ x^2 + p*x + 5 ≤ 1) → (p = 4 ∨ p = -4) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l1277_127769


namespace NUMINAMATH_CALUDE_route_length_is_200_l1277_127781

/-- Two trains traveling on a route --/
structure TrainRoute where
  length : ℝ
  train_a_time : ℝ
  train_b_time : ℝ
  meeting_distance : ℝ

/-- The specific train route from the problem --/
def problem_route : TrainRoute where
  length := 200
  train_a_time := 10
  train_b_time := 6
  meeting_distance := 75

/-- Theorem stating that the given conditions imply the route length is 200 miles --/
theorem route_length_is_200 (route : TrainRoute) :
  route.train_a_time = 10 ∧
  route.train_b_time = 6 ∧
  route.meeting_distance = 75 →
  route.length = 200 := by
  sorry

#check route_length_is_200

end NUMINAMATH_CALUDE_route_length_is_200_l1277_127781


namespace NUMINAMATH_CALUDE_safe_password_l1277_127712

def digits : List Nat := [6, 2, 5]

def largest_number (digits : List Nat) : Nat :=
  100 * (digits.maximum?.getD 0) + 
  10 * (digits.filter (· ≠ digits.maximum?.getD 0)).maximum?.getD 0 + 
  (digits.filter (· ∉ [digits.maximum?.getD 0, (digits.filter (· ≠ digits.maximum?.getD 0)).maximum?.getD 0])).sum

def smallest_number (digits : List Nat) : Nat :=
  100 * (digits.minimum?.getD 0) + 
  10 * (digits.filter (· ≠ digits.minimum?.getD 0)).minimum?.getD 0 + 
  (digits.filter (· ∉ [digits.minimum?.getD 0, (digits.filter (· ≠ digits.minimum?.getD 0)).minimum?.getD 0])).sum

theorem safe_password : 
  largest_number digits + smallest_number digits = 908 := by
  sorry

end NUMINAMATH_CALUDE_safe_password_l1277_127712


namespace NUMINAMATH_CALUDE_circular_track_circumference_l1277_127746

/-- The circumference of a circular track given two cyclists' speeds and meeting time -/
theorem circular_track_circumference (speed1 speed2 : ℝ) (time : ℝ) (h1 : speed1 = 7)
    (h2 : speed2 = 8) (h3 : time = 42) :
    speed1 * time + speed2 * time = 630 := by
  sorry

end NUMINAMATH_CALUDE_circular_track_circumference_l1277_127746


namespace NUMINAMATH_CALUDE_unique_base_sum_l1277_127732

def sum_single_digits (b : ℕ) : ℕ := 
  if b % 2 = 0 then
    b * (b - 1) / 2
  else
    (b^2 - 1) / 2

theorem unique_base_sum : 
  ∃! b : ℕ, b > 0 ∧ sum_single_digits b = 2 * b + 8 :=
sorry

end NUMINAMATH_CALUDE_unique_base_sum_l1277_127732


namespace NUMINAMATH_CALUDE_students_both_correct_l1277_127799

theorem students_both_correct (total : ℕ) (physics_correct : ℕ) (chemistry_correct : ℕ) (both_incorrect : ℕ)
  (h1 : total = 50)
  (h2 : physics_correct = 40)
  (h3 : chemistry_correct = 31)
  (h4 : both_incorrect = 4) :
  total - both_incorrect - (physics_correct + chemistry_correct - total + both_incorrect) = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_both_correct_l1277_127799


namespace NUMINAMATH_CALUDE_initial_scissors_l1277_127751

theorem initial_scissors (added : ℕ) (total : ℕ) (h1 : added = 22) (h2 : total = 76) :
  total - added = 54 := by
  sorry

end NUMINAMATH_CALUDE_initial_scissors_l1277_127751


namespace NUMINAMATH_CALUDE_waiter_tips_fraction_l1277_127754

theorem waiter_tips_fraction (salary tips : ℝ) 
  (h : tips = 0.625 * (salary + tips)) : 
  tips / salary = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_waiter_tips_fraction_l1277_127754


namespace NUMINAMATH_CALUDE_tim_nickels_l1277_127755

/-- The number of nickels Tim got for shining shoes -/
def nickels : ℕ := sorry

/-- The number of dimes Tim got for shining shoes -/
def dimes_shining : ℕ := 13

/-- The number of dimes Tim found in his tip jar -/
def dimes_tip : ℕ := 7

/-- The number of half-dollars Tim found in his tip jar -/
def half_dollars : ℕ := 9

/-- The total amount Tim got in dollars -/
def total_amount : ℚ := 665 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The value of a half-dollar in dollars -/
def half_dollar_value : ℚ := 50 / 100

theorem tim_nickels :
  nickels * nickel_value + 
  dimes_shining * dime_value + 
  dimes_tip * dime_value + 
  half_dollars * half_dollar_value = total_amount ∧
  nickels = 3 := by sorry

end NUMINAMATH_CALUDE_tim_nickels_l1277_127755


namespace NUMINAMATH_CALUDE_remainder_777_pow_777_mod_13_l1277_127757

theorem remainder_777_pow_777_mod_13 : 777^777 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_777_pow_777_mod_13_l1277_127757


namespace NUMINAMATH_CALUDE_divisibility_implication_l1277_127717

theorem divisibility_implication (x y : ℤ) : (2*x + 1) ∣ (8*y) → (2*x + 1) ∣ y := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l1277_127717


namespace NUMINAMATH_CALUDE_count_non_negative_rationals_l1277_127727

def rational_list : List ℚ := [-8, 0, -1.04, -(-3), 1/3, -|-2|]

theorem count_non_negative_rationals :
  (rational_list.filter (λ x => x ≥ 0)).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_non_negative_rationals_l1277_127727


namespace NUMINAMATH_CALUDE_shaded_square_area_ratio_l1277_127797

theorem shaded_square_area_ratio (n : ℕ) (shaded_area : ℕ) : 
  n = 5 → shaded_area = 5 → (shaded_area : ℚ) / (n^2 : ℚ) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_square_area_ratio_l1277_127797


namespace NUMINAMATH_CALUDE_consecutive_integer_fraction_minimum_l1277_127722

theorem consecutive_integer_fraction_minimum (a b : ℤ) (h1 : a = b + 1) (h2 : a > b) :
  ∀ ε > 0, ∃ a b : ℤ, a = b + 1 ∧ a > b ∧ (a + b : ℚ) / (a - b) + (a - b : ℚ) / (a + b) < 2 + ε ∧
  ∀ a' b' : ℤ, a' = b' + 1 → a' > b' → 2 ≤ (a' + b' : ℚ) / (a' - b') + (a' - b' : ℚ) / (a' + b') :=
sorry

end NUMINAMATH_CALUDE_consecutive_integer_fraction_minimum_l1277_127722


namespace NUMINAMATH_CALUDE_square_difference_dollar_l1277_127731

def dollar (a b : ℝ) : ℝ := (a - b)^2

theorem square_difference_dollar (x y : ℝ) :
  dollar (x^2 - y^2) (y^2 - x^2) = 4 * (x^4 - 2*x^2*y^2 + y^4) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_dollar_l1277_127731


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l1277_127783

/-- The amount of flour Mary has already put in the recipe -/
def flour_already_added : ℕ := 3

/-- The amount of flour Mary still needs to add to the recipe -/
def flour_to_be_added : ℕ := 6

/-- The total amount of flour required for the recipe -/
def total_flour_required : ℕ := flour_already_added + flour_to_be_added

theorem recipe_flour_amount : total_flour_required = 9 := by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l1277_127783


namespace NUMINAMATH_CALUDE_units_digit_of_17_pow_2045_l1277_127765

theorem units_digit_of_17_pow_2045 : (17^2045 : ℕ) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_17_pow_2045_l1277_127765


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1277_127778

theorem quadratic_roots_problem (k : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x^2 - 2*(k+1)*x + k^2 + 3 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  1/x₁ + 1/x₂ = 6/7 →
  k = 2 ∧ x₁^2 + x₂^2 > 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1277_127778


namespace NUMINAMATH_CALUDE_remainder_after_adding_2030_l1277_127782

theorem remainder_after_adding_2030 (m : ℤ) (h : m % 7 = 2) : (m + 2030) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2030_l1277_127782


namespace NUMINAMATH_CALUDE_illustration_project_time_l1277_127742

/-- Calculates the total time spent on an illustration project with three phases -/
def total_illustration_time (
  landscape_count : ℕ)
  (landscape_draw_time : ℝ)
  (landscape_color_reduction : ℝ)
  (landscape_enhance_time : ℝ)
  (portrait_count : ℕ)
  (portrait_draw_time : ℝ)
  (portrait_color_reduction : ℝ)
  (portrait_enhance_time : ℝ)
  (abstract_count : ℕ)
  (abstract_draw_time : ℝ)
  (abstract_color_reduction : ℝ)
  (abstract_enhance_time : ℝ) : ℝ :=
  let landscape_time := landscape_count * (landscape_draw_time + landscape_draw_time * (1 - landscape_color_reduction) + landscape_enhance_time)
  let portrait_time := portrait_count * (portrait_draw_time + portrait_draw_time * (1 - portrait_color_reduction) + portrait_enhance_time)
  let abstract_time := abstract_count * (abstract_draw_time + abstract_draw_time * (1 - abstract_color_reduction) + abstract_enhance_time)
  landscape_time + portrait_time + abstract_time

theorem illustration_project_time :
  total_illustration_time 10 2 0.3 0.75 15 3 0.25 1 20 1.5 0.4 0.5 = 193.25 := by
  sorry

end NUMINAMATH_CALUDE_illustration_project_time_l1277_127742


namespace NUMINAMATH_CALUDE_divisibility_by_five_l1277_127760

theorem divisibility_by_five (a : ℤ) : 
  5 ∣ (a^3 + 3*a + 1) ↔ a % 5 = 1 ∨ a % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l1277_127760


namespace NUMINAMATH_CALUDE_angle_properties_l1277_127756

def isObtuseAngle (α : ℝ) : Prop := 90 < α ∧ α < 180

def isSecondQuadrantAngle (β : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 + 90 < β ∧ β < k * 360 + 180

def isFirstQuadrantAngle (γ : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 < γ ∧ γ < k * 360 + 90

theorem angle_properties :
  (∀ α : ℝ, isObtuseAngle α → isSecondQuadrantAngle α) ∧
  (∃ β γ : ℝ, isSecondQuadrantAngle β ∧ isFirstQuadrantAngle γ ∧ β < γ) ∧
  (∃ δ : ℝ, 90 < δ ∧ ¬ isObtuseAngle δ) ∧
  ¬ isSecondQuadrantAngle (-165) :=
by sorry

end NUMINAMATH_CALUDE_angle_properties_l1277_127756


namespace NUMINAMATH_CALUDE_P_equals_complement_union_l1277_127771

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p | p.2 ≠ p.1}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.2 ≠ -p.1}

-- Define set P
def P : Set (ℝ × ℝ) := {p | p.2^2 ≠ p.1^2}

-- Theorem statement
theorem P_equals_complement_union :
  P = (U \ M) ∪ (U \ N) := by sorry

end NUMINAMATH_CALUDE_P_equals_complement_union_l1277_127771


namespace NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_l1277_127787

theorem ac_squared_gt_bc_squared (a b c : ℝ) : a > b → a * c^2 > b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_l1277_127787


namespace NUMINAMATH_CALUDE_modified_cube_edge_count_l1277_127749

/-- Represents a cube with smaller cubes removed from alternate corners -/
structure ModifiedCube where
  side_length : ℕ
  removed_cube_side_length : ℕ
  removed_corners : ℕ

/-- Calculates the number of edges in a modified cube -/
def edge_count (c : ModifiedCube) : ℕ :=
  12 + 3 * c.removed_corners

/-- Theorem stating that a cube of side length 4 with unit cubes removed from 4 corners has 24 edges -/
theorem modified_cube_edge_count :
  ∀ (c : ModifiedCube), 
    c.side_length = 4 ∧ 
    c.removed_cube_side_length = 1 ∧ 
    c.removed_corners = 4 → 
    edge_count c = 24 := by
  sorry

#check modified_cube_edge_count

end NUMINAMATH_CALUDE_modified_cube_edge_count_l1277_127749


namespace NUMINAMATH_CALUDE_g_fifty_l1277_127700

/-- A function g satisfying g(xy) = xg(y) for all real x and y, and g(1) = 40 -/
def g : ℝ → ℝ := sorry

/-- The property that g(xy) = xg(y) for all real x and y -/
axiom g_prop (x y : ℝ) : g (x * y) = x * g y

/-- The property that g(1) = 40 -/
axiom g_one : g 1 = 40

/-- Theorem: g(50) = 2000 -/
theorem g_fifty : g 50 = 2000 := by sorry

end NUMINAMATH_CALUDE_g_fifty_l1277_127700


namespace NUMINAMATH_CALUDE_lcm_equation_solution_l1277_127763

theorem lcm_equation_solution :
  ∀ x y : ℕ, 
    x > 0 ∧ y > 0 → 
    Nat.lcm x y = 1 + 2*x + 3*y ↔ (x = 4 ∧ y = 9) ∨ (x = 9 ∧ y = 4) := by
  sorry

end NUMINAMATH_CALUDE_lcm_equation_solution_l1277_127763


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l1277_127736

-- Define the quadratic equation
def quadratic (m : ℤ) (x : ℤ) : Prop :=
  x^2 - 2*(2*m-3)*x + 4*m^2 - 14*m + 8 = 0

-- Define the theorem
theorem quadratic_roots_theorem :
  ∀ m : ℤ, 4 < m → m < 40 →
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ quadratic m x₁ ∧ quadratic m x₂) →
  ((m = 12 ∧ ∃ x₁ x₂ : ℤ, x₁ = 26 ∧ x₂ = 16 ∧ quadratic m x₁ ∧ quadratic m x₂) ∨
   (m = 24 ∧ ∃ x₁ x₂ : ℤ, x₁ = 52 ∧ x₂ = 38 ∧ quadratic m x₁ ∧ quadratic m x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l1277_127736


namespace NUMINAMATH_CALUDE_fraction_simplification_l1277_127753

theorem fraction_simplification : 
  (20 : ℚ) / 19 * 15 / 28 * 76 / 45 = 95 / 84 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1277_127753


namespace NUMINAMATH_CALUDE_sum_seven_multiples_of_12_l1277_127768

/-- Sum of arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The sum of the first seven multiples of 12 -/
theorem sum_seven_multiples_of_12 :
  arithmetic_sum 12 12 7 = 336 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_multiples_of_12_l1277_127768


namespace NUMINAMATH_CALUDE_chef_cherries_l1277_127741

theorem chef_cherries (used_for_pie : ℕ) (left_over : ℕ) (initial : ℕ) : 
  used_for_pie = 60 → left_over = 17 → initial = used_for_pie + left_over → initial = 77 :=
by sorry

end NUMINAMATH_CALUDE_chef_cherries_l1277_127741


namespace NUMINAMATH_CALUDE_triangle_689_is_acute_l1277_127739

-- Define a triangle with sides in the ratio 6:8:9
def Triangle (t : ℝ) : Fin 3 → ℝ
| 0 => 6 * t
| 1 => 8 * t
| 2 => 9 * t

-- Define what it means for a triangle to be acute
def IsAcute (triangle : Fin 3 → ℝ) : Prop :=
  (triangle 0)^2 + (triangle 1)^2 > (triangle 2)^2 ∧
  (triangle 0)^2 + (triangle 2)^2 > (triangle 1)^2 ∧
  (triangle 1)^2 + (triangle 2)^2 > (triangle 0)^2

-- Theorem statement
theorem triangle_689_is_acute (t : ℝ) (h : t > 0) : IsAcute (Triangle t) := by
  sorry

end NUMINAMATH_CALUDE_triangle_689_is_acute_l1277_127739


namespace NUMINAMATH_CALUDE_three_cell_shapes_count_l1277_127738

/-- Represents the number of cells in a shape -/
inductive ShapeSize
| Three : ShapeSize
| Four : ShapeSize

/-- Represents a configuration of shapes -/
structure Configuration :=
  (threeCell : ℕ)
  (fourCell : ℕ)

/-- Checks if a configuration is valid -/
def isValidConfiguration (config : Configuration) : Prop :=
  3 * config.threeCell + 4 * config.fourCell = 22

/-- Checks if a configuration matches the desired solution -/
def isDesiredSolution (config : Configuration) : Prop :=
  config.threeCell = 6 ∧ config.fourCell = 1

/-- The main theorem to prove -/
theorem three_cell_shapes_count :
  ∃ (config : Configuration),
    isValidConfiguration config ∧ isDesiredSolution config :=
sorry

end NUMINAMATH_CALUDE_three_cell_shapes_count_l1277_127738


namespace NUMINAMATH_CALUDE_matthews_walking_rate_l1277_127789

/-- Proves that Matthew's walking rate is 3 km per hour given the problem conditions -/
theorem matthews_walking_rate (total_distance : ℝ) (johnny_start_delay : ℝ) (johnny_rate : ℝ) (johnny_distance : ℝ) :
  total_distance = 45 →
  johnny_start_delay = 1 →
  johnny_rate = 4 →
  johnny_distance = 24 →
  ∃ (matthews_rate : ℝ),
    matthews_rate = 3 ∧
    matthews_rate * (johnny_distance / johnny_rate + johnny_start_delay) = total_distance - johnny_distance :=
by sorry

end NUMINAMATH_CALUDE_matthews_walking_rate_l1277_127789


namespace NUMINAMATH_CALUDE_calculate_expression_l1277_127770

theorem calculate_expression : 
  let tan_60 : ℝ := Real.sqrt 3
  |2 - tan_60| - 1 + 4 + Real.sqrt 3 = 5 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_l1277_127770


namespace NUMINAMATH_CALUDE_volume_change_with_pressure_increase_l1277_127734

theorem volume_change_with_pressure_increase {P V P' V' : ℝ} (h1 : P > 0) (h2 : V > 0) :
  (P * V = P' * V') → -- inverse proportionality
  (P' = 1.2 * P) → -- 20% increase in pressure
  (V' = V * (5/6)) -- 16.67% decrease in volume
  := by sorry

end NUMINAMATH_CALUDE_volume_change_with_pressure_increase_l1277_127734


namespace NUMINAMATH_CALUDE_principal_calculation_l1277_127748

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time

/-- Problem statement -/
theorem principal_calculation (interest rate time : ℝ) 
  (h1 : interest = 4016.25)
  (h2 : rate = 0.13)
  (h3 : time = 5)
  : ∃ (principal : ℝ), simple_interest principal rate time = interest ∧ principal = 6180 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l1277_127748


namespace NUMINAMATH_CALUDE_arithmetic_geometric_equivalence_l1277_127710

def is_arithmetic_seq (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def is_geometric_seq (b : ℕ → ℕ) : Prop :=
  ∃ q : ℕ, q > 1 ∧ ∀ n, b (n + 1) = b n * q

def every_term_in (b a : ℕ → ℕ) : Prop :=
  ∀ n, ∃ m, b n = a m

theorem arithmetic_geometric_equivalence
  (a b : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_seq a d →
  is_geometric_seq b →
  a 1 = b 1 →
  a 2 = b 2 →
  (d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5) →
  every_term_in b a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_equivalence_l1277_127710


namespace NUMINAMATH_CALUDE_newspaper_conference_attendees_l1277_127786

/-- The minimum number of people attending the newspaper conference -/
def min_attendees : ℕ := 126

/-- The number of writers at the conference -/
def writers : ℕ := 35

/-- The minimum number of editors at the conference -/
def min_editors : ℕ := 39

/-- The maximum number of people who are both writers and editors -/
def max_both : ℕ := 26

/-- The number of people who are neither writers nor editors -/
def neither : ℕ := 2 * max_both

theorem newspaper_conference_attendees :
  ∀ N : ℕ,
  (N ≥ writers + min_editors - max_both + neither) →
  (N ≥ min_attendees) :=
by sorry

end NUMINAMATH_CALUDE_newspaper_conference_attendees_l1277_127786


namespace NUMINAMATH_CALUDE_min_coefficient_value_l1277_127793

theorem min_coefficient_value (a b box : ℤ) : 
  (∀ x : ℝ, (a * x + b) * (b * x + a) = 10 * x^2 + box * x + 10) →
  a ≠ b ∧ b ≠ box ∧ a ≠ box →
  box ≥ 29 :=
by sorry

end NUMINAMATH_CALUDE_min_coefficient_value_l1277_127793


namespace NUMINAMATH_CALUDE_max_annual_profit_l1277_127791

/-- Additional investment function R -/
noncomputable def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 40 then 10 * x^2 + 300 * x
  else (901 * x^2 - 9450 * x + 10000) / x

/-- Annual profit function W -/
noncomputable def W (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 40 then -10 * x^2 + 600 * x - 260
  else -x - 10000 / x + 9190

/-- Theorem stating the maximum annual profit and corresponding production volume -/
theorem max_annual_profit :
  ∃ (x : ℝ), x = 100 ∧ W x = 8990 ∧ ∀ y, W y ≤ W x :=
sorry

end NUMINAMATH_CALUDE_max_annual_profit_l1277_127791


namespace NUMINAMATH_CALUDE_bertha_descendants_no_daughters_l1277_127737

/-- Represents a person in Bertha's family tree -/
inductive Person
| bertha : Person
| child : Person → Person
| grandchild : Person → Person
| greatgrandchild : Person → Person

/-- Represents the gender of a person -/
inductive Gender
| male
| female

/-- Function to determine the gender of a person -/
def gender : Person → Gender
| Person.bertha => Gender.female
| _ => sorry

/-- Function to count the number of daughters a person has -/
def daughterCount : Person → Nat
| Person.bertha => 7
| _ => sorry

/-- Function to count the number of sons a person has -/
def sonCount : Person → Nat
| Person.bertha => 3
| _ => sorry

/-- Function to count the total number of female descendants of a person -/
def femaleDescendantCount : Person → Nat
| Person.bertha => 40
| _ => sorry

/-- Function to determine if a person has exactly three daughters -/
def hasThreeDaughters : Person → Bool
| _ => sorry

/-- Function to count the number of descendants (including the person) who have no daughters -/
def descendantsWithNoDaughters : Person → Nat
| _ => sorry

/-- Theorem stating that the number of Bertha's descendants with no daughters is 28 -/
theorem bertha_descendants_no_daughters :
  descendantsWithNoDaughters Person.bertha = 28 := by sorry

end NUMINAMATH_CALUDE_bertha_descendants_no_daughters_l1277_127737


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1277_127724

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ s = -b / a) :=
sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 + 2000 * x - 2001
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ s = -2000) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1277_127724


namespace NUMINAMATH_CALUDE_product_arrangement_count_l1277_127795

/-- The number of products to arrange -/
def n : ℕ := 5

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldr (· * ·) 1

/-- The number of arrangements with A and B together -/
def arrangementsWithABTogether : ℕ := 2 * factorial (n - 1)

/-- The number of arrangements with C and D together -/
def arrangementsWithCDTogether : ℕ := 2 * 2 * factorial (n - 2)

/-- The total number of valid arrangements -/
def validArrangements : ℕ := arrangementsWithABTogether - arrangementsWithCDTogether

theorem product_arrangement_count : validArrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_product_arrangement_count_l1277_127795


namespace NUMINAMATH_CALUDE_inequality_proof_l1277_127794

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b) ≥ (a + b + c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1277_127794


namespace NUMINAMATH_CALUDE_prob_odd_die_roll_l1277_127759

/-- The number of possible outcomes when rolling a die -/
def total_outcomes : ℕ := 6

/-- The number of favorable outcomes (odd numbers) when rolling a die -/
def favorable_outcomes : ℕ := 3

/-- The probability of an event in a finite sample space -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

/-- Theorem: The probability of rolling an odd number on a standard six-sided die is 1/2 -/
theorem prob_odd_die_roll : probability favorable_outcomes total_outcomes = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_die_roll_l1277_127759


namespace NUMINAMATH_CALUDE_exists_solution_l1277_127758

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + Real.exp (x - a)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem exists_solution (a : ℝ) : 
  (∃ x₀ : ℝ, f a x₀ - g a x₀ = 3) → a = -Real.log 2 - 1 := by sorry

end NUMINAMATH_CALUDE_exists_solution_l1277_127758


namespace NUMINAMATH_CALUDE_ellipse_and_chord_l1277_127719

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem about an ellipse and a chord -/
theorem ellipse_and_chord 
  (C : Ellipse) 
  (h_ecc : C.a * C.a - C.b * C.b = (C.a * C.a) / 4) 
  (h_point : (2 : ℝ) * (2 : ℝ) / (C.a * C.a) + (-3 : ℝ) * (-3 : ℝ) / (C.b * C.b) = 1)
  (M : Point) 
  (h_M : M.x = -1 ∧ M.y = 2) :
  (∃ (D : Ellipse), D.a * D.a = 16 ∧ D.b * D.b = 12) ∧
  (∃ (l : Line), l.a = 3 ∧ l.b = -8 ∧ l.c = 19) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_chord_l1277_127719


namespace NUMINAMATH_CALUDE_marble_probability_l1277_127775

/-- The number of green marbles -/
def green_marbles : ℕ := 7

/-- The number of purple marbles -/
def purple_marbles : ℕ := 5

/-- The total number of marbles -/
def total_marbles : ℕ := green_marbles + purple_marbles

/-- The number of trials -/
def num_trials : ℕ := 8

/-- The number of successful outcomes (choosing green marbles) -/
def num_success : ℕ := 3

/-- The probability of choosing a green marble in a single trial -/
def p : ℚ := green_marbles / total_marbles

/-- The probability of choosing a purple marble in a single trial -/
def q : ℚ := purple_marbles / total_marbles

/-- The binomial probability of choosing exactly 3 green marbles in 8 trials -/
def binomial_prob : ℚ := (Nat.choose num_trials num_success : ℚ) * p ^ num_success * q ^ (num_trials - num_success)

theorem marble_probability : binomial_prob = 9378906 / 67184015 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l1277_127775


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1277_127796

theorem complex_equation_solution (z : ℂ) : (3 - 4*I)*z = 25 → z = 3 + 4*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1277_127796


namespace NUMINAMATH_CALUDE_unique_students_count_l1277_127703

theorem unique_students_count (orchestra band choir : ℕ) 
  (orchestra_band orchestra_choir band_choir all_three : ℕ) :
  orchestra = 25 →
  band = 40 →
  choir = 30 →
  orchestra_band = 5 →
  orchestra_choir = 6 →
  band_choir = 4 →
  all_three = 2 →
  orchestra + band + choir - (orchestra_band + orchestra_choir + band_choir) + all_three = 82 :=
by sorry

end NUMINAMATH_CALUDE_unique_students_count_l1277_127703


namespace NUMINAMATH_CALUDE_exists_bound_for_digit_sum_of_factorial_l1277_127764

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The statement to be proved -/
theorem exists_bound_for_digit_sum_of_factorial :
  ∃ b : ℕ, ∀ n : ℕ, n > b → sum_of_digits (n.factorial) ≥ 10^100 := by
  sorry

end NUMINAMATH_CALUDE_exists_bound_for_digit_sum_of_factorial_l1277_127764


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l1277_127716

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, (2 * x - m) / (x + 1) = 1 ∧ x < 0) ↔ (m < -1 ∧ m ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l1277_127716


namespace NUMINAMATH_CALUDE_trig_identity_l1277_127714

theorem trig_identity (α β : Real) 
  (h : (Real.sin β)^4 / (Real.sin α)^2 + (Real.cos β)^4 / (Real.cos α)^2 = 1) :
  ∃ x, (Real.cos α)^4 / (Real.cos β)^2 + (Real.sin α)^4 / (Real.sin β)^2 = x ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_trig_identity_l1277_127714


namespace NUMINAMATH_CALUDE_mod_seven_equivalence_l1277_127767

theorem mod_seven_equivalence : (41^1723 - 18^1723) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_seven_equivalence_l1277_127767


namespace NUMINAMATH_CALUDE_negative_integer_equation_solution_l1277_127745

theorem negative_integer_equation_solution :
  ∃ (M : ℤ), (M < 0) ∧ (2 * M^2 + M = 12) → M = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_equation_solution_l1277_127745


namespace NUMINAMATH_CALUDE_goat_max_distance_l1277_127798

theorem goat_max_distance (center : ℝ × ℝ) (radius : ℝ) :
  center = (6, 8) →
  radius = 15 →
  let dist_to_center := Real.sqrt ((center.1 - 0)^2 + (center.2 - 0)^2)
  let max_distance := dist_to_center + radius
  max_distance = 25 := by sorry

end NUMINAMATH_CALUDE_goat_max_distance_l1277_127798


namespace NUMINAMATH_CALUDE_line_equation_theorem_l1277_127774

-- Define the line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def has_slope (l : Line) (m : ℝ) : Prop :=
  l.a ≠ 0 ∧ -l.b / l.a = m

def triangle_area (l : Line) (area : ℝ) : Prop :=
  l.c ≠ 0 ∧ abs (l.c / l.a) * abs (l.c / l.b) / 2 = area

def passes_through (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

def equal_absolute_intercepts (l : Line) : Prop :=
  abs (l.c / l.a) = abs (l.c / l.b)

-- Define the theorem
theorem line_equation_theorem (l : Line) :
  has_slope l (3/4) ∧
  triangle_area l 6 ∧
  passes_through l 4 (-3) ∧
  equal_absolute_intercepts l →
  (l.a = 1 ∧ l.b = 1 ∧ l.c = -1) ∨
  (l.a = 1 ∧ l.b = -1 ∧ l.c = 7) ∨
  (l.a = 3 ∧ l.b = 4 ∧ l.c = 0) :=
sorry

end NUMINAMATH_CALUDE_line_equation_theorem_l1277_127774


namespace NUMINAMATH_CALUDE_rhombus_diagonals_not_always_equal_l1277_127702

-- Define a rhombus
structure Rhombus :=
  (side_length : ℝ)
  (diagonal1 : ℝ)
  (diagonal2 : ℝ)
  (side_length_positive : side_length > 0)
  (diagonals_positive : diagonal1 > 0 ∧ diagonal2 > 0)

-- State the theorem
theorem rhombus_diagonals_not_always_equal :
  ∃ (r : Rhombus), r.diagonal1 ≠ r.diagonal2 :=
sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_not_always_equal_l1277_127702


namespace NUMINAMATH_CALUDE_complex_base_representation_exists_unique_integer_representable_in_base_neg4_plus_i_l1277_127784

/-- Representation of a complex number in a complex base -n+i --/
structure ComplexBaseRepresentation (n : ℕ+) where
  coeffs : Fin 4 → Fin 257
  nonzero_lead : coeffs 3 ≠ 0

/-- The value represented by a ComplexBaseRepresentation --/
def value (n : ℕ+) (rep : ComplexBaseRepresentation n) : ℂ :=
  (rep.coeffs 3 : ℂ) * (-n + Complex.I)^3 +
  (rep.coeffs 2 : ℂ) * (-n + Complex.I)^2 +
  (rep.coeffs 1 : ℂ) * (-n + Complex.I) +
  (rep.coeffs 0 : ℂ)

/-- Theorem stating the existence and uniqueness of the representation --/
theorem complex_base_representation_exists_unique (n : ℕ+) (z : ℂ) 
  (h : ∃ (r s : ℤ), z = r + s * Complex.I) :
  ∃! (rep : ComplexBaseRepresentation n), value n rep = z :=
sorry

/-- Theorem stating that for base -4+i, there exist integers representable in four digits --/
theorem integer_representable_in_base_neg4_plus_i :
  ∃ (k : ℤ) (rep : ComplexBaseRepresentation 4),
    value 4 rep = k ∧ k = (value 4 rep).re :=
sorry

end NUMINAMATH_CALUDE_complex_base_representation_exists_unique_integer_representable_in_base_neg4_plus_i_l1277_127784


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_two_zeros_l1277_127704

/-- The number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers with fewer than two zeros -/
def fewer_than_two_zeros : ℕ := 826686

/-- The number of 6-digit numbers with at least two zeros -/
def at_least_two_zeros : ℕ := total_six_digit_numbers - fewer_than_two_zeros

theorem six_digit_numbers_with_two_zeros :
  at_least_two_zeros = 73314 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_two_zeros_l1277_127704


namespace NUMINAMATH_CALUDE_recurring_decimal_equals_two_fifteenths_l1277_127701

-- Define the recurring decimal 0.333...
def recurring_third : ℚ := 1/3

-- Define the recurring decimal 0.1333...
def recurring_decimal : ℚ := 0.1333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333

-- State the theorem
theorem recurring_decimal_equals_two_fifteenths 
  (h : recurring_third = 1/3) : 
  recurring_decimal = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_equals_two_fifteenths_l1277_127701


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l1277_127785

-- Define the complex number z
def z : ℂ := sorry

-- State the theorem
theorem z_in_first_quadrant :
  (2 - Complex.I) * z = 1 + Complex.I →
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l1277_127785


namespace NUMINAMATH_CALUDE_wages_comparison_l1277_127706

theorem wages_comparison (erica robin charles : ℝ) 
  (h1 : robin = 1.3 * erica) 
  (h2 : charles = 1.23076923076923077 * robin) : 
  charles = 1.6 * erica := by
sorry

end NUMINAMATH_CALUDE_wages_comparison_l1277_127706
