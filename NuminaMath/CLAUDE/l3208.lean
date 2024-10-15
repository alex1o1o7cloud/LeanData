import Mathlib

namespace NUMINAMATH_CALUDE_triangle_inradius_l3208_320890

/-- Given a triangle with perimeter 28 cm and area 35 cm², prove that its inradius is 2.5 cm. -/
theorem triangle_inradius (P : ℝ) (A : ℝ) (r : ℝ) : 
  P = 28 → A = 35 → A = r * (P / 2) → r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l3208_320890


namespace NUMINAMATH_CALUDE_line_slope_calculation_l3208_320815

/-- Given a line in the xy-plane with y-intercept 20 and passing through the point (150, 600),
    its slope is equal to 580/150. -/
theorem line_slope_calculation (line : Set (ℝ × ℝ)) : 
  (∀ p ∈ line, ∃ m b : ℝ, p.2 = m * p.1 + b) →  -- Line equation
  (0, 20) ∈ line →                              -- y-intercept condition
  (150, 600) ∈ line →                           -- Point condition
  ∃ m : ℝ, m = 580 / 150 ∧                      -- Slope existence and value
    ∀ (x y : ℝ), (x, y) ∈ line → y = m * x + 20 -- Line equation with calculated slope
  := by sorry

end NUMINAMATH_CALUDE_line_slope_calculation_l3208_320815


namespace NUMINAMATH_CALUDE_travel_distance_ratio_l3208_320801

theorem travel_distance_ratio (total_distance train_distance : ℝ)
  (h1 : total_distance = 500)
  (h2 : train_distance = 300)
  (h3 : ∃ bus_distance cab_distance : ℝ,
    total_distance = train_distance + bus_distance + cab_distance ∧
    cab_distance = (1/3) * bus_distance) :
  ∃ bus_distance : ℝ, bus_distance / train_distance = 1/2 :=
sorry

end NUMINAMATH_CALUDE_travel_distance_ratio_l3208_320801


namespace NUMINAMATH_CALUDE_abc_sum_product_bound_l3208_320855

theorem abc_sum_product_bound (a b c : ℝ) (h : a + b + c = 1) :
  ∃ (M : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (a' b' c' : ℝ),
    a' + b' + c' = 1 ∧ 
    ab + ac + bc ≤ 1/2 ∧
    a' * b' + a' * c' + b' * c' < -M + ε :=
sorry

end NUMINAMATH_CALUDE_abc_sum_product_bound_l3208_320855


namespace NUMINAMATH_CALUDE_min_toothpicks_removal_l3208_320806

/-- Represents a triangular figure made of toothpicks -/
structure TriangularFigure where
  toothpicks : ℕ
  triangles : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : TriangularFigure) : ℕ :=
  15

/-- Theorem: For a triangular figure with 40 toothpicks and at least 35 triangles,
    the minimum number of toothpicks to remove to eliminate all triangles is 15 -/
theorem min_toothpicks_removal (figure : TriangularFigure) 
    (h1 : figure.toothpicks = 40) 
    (h2 : figure.triangles ≥ 35) : 
  min_toothpicks_to_remove figure = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_min_toothpicks_removal_l3208_320806


namespace NUMINAMATH_CALUDE_calculate_expression_l3208_320838

theorem calculate_expression : (121^2 - 110^2 + 11) / 10 = 255.2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3208_320838


namespace NUMINAMATH_CALUDE_folded_rectangle_theorem_l3208_320891

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a rectangle with vertices A, B, C, D -/
structure Rectangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Represents the folded state of the rectangle -/
structure FoldedRectangle :=
  (rect : Rectangle)
  (E : Point)
  (F : Point)

/-- Given a folded rectangle, returns the measure of Angle 1 in degrees -/
def angle1 (fr : FoldedRectangle) : ℝ := sorry

/-- Given a folded rectangle, returns the measure of Angle 2 in degrees -/
def angle2 (fr : FoldedRectangle) : ℝ := sorry

/-- Predicate to check if a point is on a line segment -/
def isOnSegment (P Q R : Point) : Prop := sorry

/-- Predicate to check if two triangles are congruent -/
def areCongruentTriangles (ABC DEF : Point × Point × Point) : Prop := sorry

theorem folded_rectangle_theorem (fr : FoldedRectangle) :
  isOnSegment fr.rect.A fr.E fr.rect.B ∧
  areCongruentTriangles (fr.rect.D, fr.rect.C, fr.F) (fr.rect.D, fr.E, fr.F) ∧
  angle1 fr = 22 →
  angle2 fr = 44 :=
sorry

end NUMINAMATH_CALUDE_folded_rectangle_theorem_l3208_320891


namespace NUMINAMATH_CALUDE_fractional_to_polynomial_equivalence_l3208_320812

theorem fractional_to_polynomial_equivalence (x : ℝ) (h : x ≠ 2) :
  (x / (x - 2) + 2 = 1 / (2 - x)) ↔ (x + 2 * (x - 2) = -1) :=
sorry

end NUMINAMATH_CALUDE_fractional_to_polynomial_equivalence_l3208_320812


namespace NUMINAMATH_CALUDE_car_speed_problem_l3208_320860

/-- Proves that car R's average speed is 50 miles per hour given the problem conditions -/
theorem car_speed_problem (distance : ℝ) (time_difference : ℝ) (speed_difference : ℝ) :
  distance = 800 ∧ 
  time_difference = 2 ∧ 
  speed_difference = 10 →
  ∃ (speed_r : ℝ),
    speed_r > 0 ∧
    distance / speed_r - time_difference = distance / (speed_r + speed_difference) ∧
    speed_r = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3208_320860


namespace NUMINAMATH_CALUDE_flag_rectangle_ratio_l3208_320863

/-- Given a rectangle with side lengths in ratio 3:5, divided into four equal area rectangles,
    the ratio of the shorter side to the longer side of one of these rectangles is 4:15 -/
theorem flag_rectangle_ratio :
  ∀ (k : ℝ), k > 0 →
  let flag_width := 5 * k
  let flag_height := 3 * k
  let small_rect_area := (flag_width * flag_height) / 4
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
    x * y = small_rect_area ∧
    3 * y = k ∧
    5 * y = x ∧
    y / x = 4 / 15 :=
by sorry

end NUMINAMATH_CALUDE_flag_rectangle_ratio_l3208_320863


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l3208_320848

/-- Proves that adding 1.8 liters of pure alcohol to a 6-liter solution
    that is 35% alcohol will result in a solution that is 50% alcohol. -/
theorem alcohol_solution_proof :
  let initial_volume : ℝ := 6
  let initial_concentration : ℝ := 0.35
  let target_concentration : ℝ := 0.50
  let added_alcohol : ℝ := 1.8

  let final_volume : ℝ := initial_volume + added_alcohol
  let initial_alcohol : ℝ := initial_volume * initial_concentration
  let final_alcohol : ℝ := initial_alcohol + added_alcohol

  (final_alcohol / final_volume) = target_concentration :=
by sorry


end NUMINAMATH_CALUDE_alcohol_solution_proof_l3208_320848


namespace NUMINAMATH_CALUDE_revenue_decrease_l3208_320819

theorem revenue_decrease (tax_reduction : Real) (consumption_increase : Real) :
  tax_reduction = 0.24 →
  consumption_increase = 0.12 →
  let new_tax_rate := 1 - tax_reduction
  let new_consumption := 1 + consumption_increase
  let revenue_change := 1 - (new_tax_rate * new_consumption)
  revenue_change = 0.1488 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_l3208_320819


namespace NUMINAMATH_CALUDE_remaining_budget_for_accessories_l3208_320899

def total_budget : ℕ := 250
def frame_cost : ℕ := 85
def front_wheel_cost : ℕ := 35
def rear_wheel_cost : ℕ := 40
def seat_cost : ℕ := 25
def handlebar_tape_cost : ℕ := 15
def water_bottle_cage_cost : ℕ := 10
def bike_lock_cost : ℕ := 20
def future_expenses : ℕ := 10

def total_expenses : ℕ :=
  frame_cost + front_wheel_cost + rear_wheel_cost + seat_cost +
  handlebar_tape_cost + water_bottle_cage_cost + bike_lock_cost + future_expenses

theorem remaining_budget_for_accessories :
  total_budget - total_expenses = 10 := by sorry

end NUMINAMATH_CALUDE_remaining_budget_for_accessories_l3208_320899


namespace NUMINAMATH_CALUDE_square_graph_triangles_l3208_320869

/-- A planar graph formed by a square with interior points -/
structure SquareGraph where
  /-- The number of interior points in the square -/
  interior_points : ℕ
  /-- The total number of vertices in the graph -/
  vertices : ℕ
  /-- The total number of edges in the graph -/
  edges : ℕ
  /-- The total number of faces in the graph (including the exterior face) -/
  faces : ℕ
  /-- The condition that the graph is formed by a square with interior points -/
  square_condition : vertices = interior_points + 4
  /-- The condition that all regions except the exterior are triangles -/
  triangle_condition : 2 * edges = 3 * (faces - 1) + 4
  /-- Euler's formula for planar graphs -/
  euler_formula : vertices - edges + faces = 2

/-- The theorem stating the number of triangles in the specific square graph -/
theorem square_graph_triangles (g : SquareGraph) (h : g.interior_points = 20) :
  g.faces - 1 = 42 := by
  sorry

end NUMINAMATH_CALUDE_square_graph_triangles_l3208_320869


namespace NUMINAMATH_CALUDE_interest_rate_proof_l3208_320808

/-- Proves that given specific conditions, the interest rate is 18% --/
theorem interest_rate_proof (principal : ℝ) (time : ℝ) (interest_difference : ℝ) : 
  principal = 4000 →
  time = 2 →
  interest_difference = 480 →
  (principal * time * (18 / 100)) = (principal * time * (12 / 100) + interest_difference) :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l3208_320808


namespace NUMINAMATH_CALUDE_expression_defined_iff_l3208_320823

theorem expression_defined_iff (a : ℝ) :
  (∃ x : ℝ, x = (Real.sqrt (a + 1)) / (a - 2)) ↔ (a ≥ -1 ∧ a ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_defined_iff_l3208_320823


namespace NUMINAMATH_CALUDE_polynomial_value_at_n_plus_one_l3208_320879

theorem polynomial_value_at_n_plus_one (n : ℕ) (p : ℝ → ℝ) :
  (∀ k : ℕ, k ≤ n → p k = k / (k + 1)) →
  p (n + 1) = if n % 2 = 1 then 1 else n / (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_n_plus_one_l3208_320879


namespace NUMINAMATH_CALUDE_ann_shorts_purchase_l3208_320847

/-- Calculates the maximum number of shorts Ann can buy -/
def max_shorts (total_spent : ℕ) (shoe_cost : ℕ) (shorts_cost : ℕ) (num_tops : ℕ) : ℕ :=
  ((total_spent - shoe_cost) / shorts_cost)

theorem ann_shorts_purchase :
  let total_spent := 75
  let shoe_cost := 20
  let shorts_cost := 7
  let num_tops := 4
  max_shorts total_spent shoe_cost shorts_cost num_tops = 7 := by
  sorry

#eval max_shorts 75 20 7 4

end NUMINAMATH_CALUDE_ann_shorts_purchase_l3208_320847


namespace NUMINAMATH_CALUDE_tan_theta_three_expression_l3208_320841

theorem tan_theta_three_expression (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ ^ 2) = (11 * Real.sqrt 10 - 101) / 33 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_three_expression_l3208_320841


namespace NUMINAMATH_CALUDE_solve_sleep_problem_l3208_320843

def sleep_problem (connor_sleep : ℕ) : Prop :=
  let luke_sleep := connor_sleep + 2
  let emma_sleep := connor_sleep - 1
  let puppy_sleep := luke_sleep * 2
  connor_sleep = 6 →
  connor_sleep + luke_sleep + emma_sleep + puppy_sleep = 35

theorem solve_sleep_problem :
  sleep_problem 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_sleep_problem_l3208_320843


namespace NUMINAMATH_CALUDE_green_balls_count_l3208_320862

theorem green_balls_count (blue_count : ℕ) (total_count : ℕ) 
  (h1 : blue_count = 8)
  (h2 : (blue_count : ℚ) / total_count = 1 / 3) :
  total_count - blue_count = 16 := by
  sorry

end NUMINAMATH_CALUDE_green_balls_count_l3208_320862


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3208_320803

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3208_320803


namespace NUMINAMATH_CALUDE_race_conditions_satisfied_l3208_320822

/-- The speed of Xiao Ying in meters per second -/
def xiao_ying_speed : ℝ := 4

/-- The speed of Xiao Liang in meters per second -/
def xiao_liang_speed : ℝ := 6

/-- Theorem stating that the given speeds satisfy the race conditions -/
theorem race_conditions_satisfied : 
  (5 * xiao_ying_speed + 10 = 5 * xiao_liang_speed) ∧ 
  (6 * xiao_ying_speed = 4 * xiao_liang_speed) := by
  sorry

end NUMINAMATH_CALUDE_race_conditions_satisfied_l3208_320822


namespace NUMINAMATH_CALUDE_f_of_3_equals_8_l3208_320898

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x - 1) + 2

-- State the theorem
theorem f_of_3_equals_8 : f 3 = 8 := by sorry

end NUMINAMATH_CALUDE_f_of_3_equals_8_l3208_320898


namespace NUMINAMATH_CALUDE_system_solution_l3208_320827

theorem system_solution : ∃! (x y : ℝ), x - y = 2 ∧ 2*x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3208_320827


namespace NUMINAMATH_CALUDE_parabola_focus_at_triangle_centroid_l3208_320878

/-- Given a triangle ABC with vertices A(-1,2), B(3,4), and C(4,-6),
    and a parabola y^2 = ax with focus at the centroid of ABC,
    prove that a = 8. -/
theorem parabola_focus_at_triangle_centroid :
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (3, 4)
  let C : ℝ × ℝ := (4, -6)
  let centroid : ℝ × ℝ := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  ∀ a : ℝ, (∀ x y : ℝ, y^2 = a*x → (x = a/4 ↔ (x, y) = centroid)) → a = 8 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_at_triangle_centroid_l3208_320878


namespace NUMINAMATH_CALUDE_cars_between_black_and_white_l3208_320872

theorem cars_between_black_and_white :
  ∀ (n : ℕ) (black_pos_right : ℕ) (white_pos_left : ℕ),
    n = 20 →
    black_pos_right = 16 →
    white_pos_left = 11 →
    (n - black_pos_right) - (white_pos_left - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cars_between_black_and_white_l3208_320872


namespace NUMINAMATH_CALUDE_points_four_units_away_l3208_320817

theorem points_four_units_away (P : ℝ) : 
  P = -3 → {x : ℝ | |x - P| = 4} = {1, -7} := by sorry

end NUMINAMATH_CALUDE_points_four_units_away_l3208_320817


namespace NUMINAMATH_CALUDE_round_73_26_repeating_l3208_320876

/-- Represents a repeating decimal number -/
structure RepeatingDecimal where
  integerPart : ℕ
  nonRepeatingPart : ℕ
  repeatingPart : ℕ

/-- Rounds a RepeatingDecimal to the nearest hundredth -/
def roundToHundredth (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The specific number 73.2626... -/
def number : RepeatingDecimal :=
  { integerPart := 73,
    nonRepeatingPart := 26,
    repeatingPart := 26 }

theorem round_73_26_repeating :
  roundToHundredth number = 73.26 :=
sorry

end NUMINAMATH_CALUDE_round_73_26_repeating_l3208_320876


namespace NUMINAMATH_CALUDE_longest_diagonal_twice_side_l3208_320850

/-- Regular octagon with side length a, shortest diagonal b, and longest diagonal c -/
structure RegularOctagon where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a

/-- Theorem: In a regular octagon, the longest diagonal is twice the side length -/
theorem longest_diagonal_twice_side (octagon : RegularOctagon) : octagon.c = 2 * octagon.a := by
  sorry

#check longest_diagonal_twice_side

end NUMINAMATH_CALUDE_longest_diagonal_twice_side_l3208_320850


namespace NUMINAMATH_CALUDE_henry_chore_earnings_l3208_320814

theorem henry_chore_earnings : ∃ (earned : ℕ), 
  (5 + earned + 13 = 20) ∧ earned = 2 := by
  sorry

end NUMINAMATH_CALUDE_henry_chore_earnings_l3208_320814


namespace NUMINAMATH_CALUDE_mixed_doubles_pairing_methods_l3208_320826

-- Define the number of male and female players
def num_male_players : ℕ := 5
def num_female_players : ℕ := 4

-- Define the number of players to be selected for each gender
def male_players_to_select : ℕ := 2
def female_players_to_select : ℕ := 2

-- Define the total number of pairing methods
def total_pairing_methods : ℕ := 120

-- Theorem statement
theorem mixed_doubles_pairing_methods :
  (Nat.choose num_male_players male_players_to_select) *
  (Nat.choose num_female_players female_players_to_select) * 2 =
  total_pairing_methods := by
  sorry


end NUMINAMATH_CALUDE_mixed_doubles_pairing_methods_l3208_320826


namespace NUMINAMATH_CALUDE_f_composition_result_l3208_320834

noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then -z^2 else z^2

theorem f_composition_result :
  f (f (f (f (1 + 2*I)))) = 503521 + 420000*I :=
by sorry

end NUMINAMATH_CALUDE_f_composition_result_l3208_320834


namespace NUMINAMATH_CALUDE_find_m_value_l3208_320870

/-- Given two functions f and g, prove the value of m when f(5) - g(5) = 20 -/
theorem find_m_value (f g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = 4*x^2 + 3*x + 5) →
  (∀ x, g x = x^2 - m*x - 9) →
  f 5 - g 5 = 20 →
  m = -16.8 := by
sorry

end NUMINAMATH_CALUDE_find_m_value_l3208_320870


namespace NUMINAMATH_CALUDE_parallelepipeds_crossed_diagonal_count_l3208_320802

/-- The edge length of the cube -/
def cube_edge : ℕ := 90

/-- The dimensions of the rectangular parallelepiped -/
def parallelepiped_dims : Fin 3 → ℕ
| 0 => 2
| 1 => 3
| 2 => 5

/-- The number of parallelepipeds that fit along each dimension of the cube -/
def parallelepipeds_per_dim (i : Fin 3) : ℕ := cube_edge / parallelepiped_dims i

/-- The total number of parallelepipeds that fit in the cube -/
def total_parallelepipeds : ℕ := (parallelepipeds_per_dim 0) * (parallelepipeds_per_dim 1) * (parallelepipeds_per_dim 2)

/-- The number of parallelepipeds crossed by a space diagonal of the cube -/
def parallelepipeds_crossed_by_diagonal : ℕ := 65

theorem parallelepipeds_crossed_diagonal_count :
  parallelepipeds_crossed_by_diagonal = 65 :=
sorry

end NUMINAMATH_CALUDE_parallelepipeds_crossed_diagonal_count_l3208_320802


namespace NUMINAMATH_CALUDE_total_profit_is_30000_l3208_320895

/-- Represents the profit distribution problem -/
structure ProfitProblem where
  total_subscription : ℕ
  a_more_than_b : ℕ
  b_more_than_c : ℕ
  b_profit : ℕ

/-- Calculate the total profit given the problem parameters -/
def calculate_total_profit (p : ProfitProblem) : ℕ :=
  sorry

/-- Theorem stating that the total profit is 30000 given the specific problem parameters -/
theorem total_profit_is_30000 :
  let p : ProfitProblem := {
    total_subscription := 50000,
    a_more_than_b := 4000,
    b_more_than_c := 5000,
    b_profit := 10200
  }
  calculate_total_profit p = 30000 := by sorry

end NUMINAMATH_CALUDE_total_profit_is_30000_l3208_320895


namespace NUMINAMATH_CALUDE_collinearity_condition_perpendicularity_condition_l3208_320833

-- Define the points as functions of a
def A (a : ℝ) : ℝ × ℝ := (1, -2*a)
def B (a : ℝ) : ℝ × ℝ := (2, a)
def C (a : ℝ) : ℝ × ℝ := (2+a, 0)
def D (a : ℝ) : ℝ × ℝ := (2*a, 1)

-- Define collinearity of three points
def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

-- Define perpendicularity of two lines
def perpendicular (p1 q1 p2 q2 : ℝ × ℝ) : Prop :=
  (q1.2 - p1.2) * (q2.2 - p2.2) = -(q1.1 - p1.1) * (q2.1 - p2.1)

-- Theorem 1: Collinearity condition
theorem collinearity_condition :
  ∀ a : ℝ, collinear (A a) (B a) (C a) ↔ a = -1/3 :=
sorry

-- Theorem 2: Perpendicularity condition
theorem perpendicularity_condition :
  ∀ a : ℝ, perpendicular (A a) (B a) (C a) (D a) ↔ a = 1/2 :=
sorry

end NUMINAMATH_CALUDE_collinearity_condition_perpendicularity_condition_l3208_320833


namespace NUMINAMATH_CALUDE_afternoon_to_morning_ratio_l3208_320816

def total_pears : ℕ := 420
def afternoon_pears : ℕ := 280

theorem afternoon_to_morning_ratio :
  let morning_pears := total_pears - afternoon_pears
  (afternoon_pears : ℚ) / morning_pears = 2 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_to_morning_ratio_l3208_320816


namespace NUMINAMATH_CALUDE_family_average_age_unchanged_l3208_320844

theorem family_average_age_unchanged 
  (initial_members : ℕ) 
  (initial_avg_age : ℝ) 
  (years_passed : ℕ) 
  (baby_age : ℝ) 
  (h1 : initial_members = 5)
  (h2 : initial_avg_age = 17)
  (h3 : years_passed = 3)
  (h4 : baby_age = 2) : 
  initial_avg_age = 
    (initial_members * (initial_avg_age + years_passed) + baby_age) / (initial_members + 1) := by
  sorry

#check family_average_age_unchanged

end NUMINAMATH_CALUDE_family_average_age_unchanged_l3208_320844


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3208_320861

theorem triangle_angle_measure (a b : ℝ) (area : ℝ) (h1 : a = 5) (h2 : b = 8) (h3 : area = 10) :
  ∃ (C : ℝ), (C = π / 6 ∨ C = 5 * π / 6) ∧ 
  (1 / 2 * a * b * Real.sin C = area) ∧ 
  (0 < C) ∧ (C < π) := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3208_320861


namespace NUMINAMATH_CALUDE_inequality_proof_l3208_320873

theorem inequality_proof (a b c : ℝ) 
  (ha : a = Real.tan (23 * π / 180) / (1 - Real.tan (23 * π / 180)^2))
  (hb : b = 2 * Real.sin (13 * π / 180) * Real.cos (13 * π / 180))
  (hc : c = Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)) :
  c < b ∧ b < a :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3208_320873


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_l3208_320889

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the interior angles of a triangle
def interior_angles (t : Triangle) : ℝ := sorry

-- Theorem: The sum of interior angles of a triangle is 180°
theorem sum_of_interior_angles (t : Triangle) : interior_angles t = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_interior_angles_l3208_320889


namespace NUMINAMATH_CALUDE_min_value_constraint_l3208_320807

theorem min_value_constraint (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  x^2 + 8*x*y + 25*y^2 + 16*y*z + 9*z^2 ≥ 403/9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_constraint_l3208_320807


namespace NUMINAMATH_CALUDE_binomial_and_permutation_l3208_320858

theorem binomial_and_permutation :
  (Nat.choose 8 5 = 56) ∧
  (Nat.factorial 5 / Nat.factorial 2 = 60) := by
  sorry

end NUMINAMATH_CALUDE_binomial_and_permutation_l3208_320858


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3208_320804

def polynomial (x : ℝ) : ℝ := 5 * (2 * x^8 - 3 * x^5 + 9) + 6 * (x^6 + 4 * x^3 - 6)

theorem sum_of_coefficients : (polynomial 1) = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3208_320804


namespace NUMINAMATH_CALUDE_betty_payment_l3208_320859

-- Define the given conditions
def doug_age : ℕ := 40
def sum_of_ages : ℕ := 90
def num_packs : ℕ := 20

-- Define Betty's age
def betty_age : ℕ := sum_of_ages - doug_age

-- Define the cost of a pack of nuts
def pack_cost : ℕ := 2 * betty_age

-- Theorem to prove
theorem betty_payment : betty_age * num_packs * 2 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_betty_payment_l3208_320859


namespace NUMINAMATH_CALUDE_line_perpendicular_to_parallel_line_l3208_320853

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_parallel_line
  (a b : Line) (α : Plane)
  (h1 : perpendicular a α)
  (h2 : parallel b α) :
  perpendicularLines a b :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_parallel_line_l3208_320853


namespace NUMINAMATH_CALUDE_f_increasing_f_two_zeros_l3208_320810

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * abs (x + 1) + a * x

-- Statement 1: f(x) is increasing when a > 2
theorem f_increasing (a : ℝ) (h : a > 2) : 
  StrictMono (f a) := by sorry

-- Statement 2: f(x) has two zeros iff a ∈ (0, 2)
theorem f_two_zeros (a : ℝ) : 
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ 0 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_f_increasing_f_two_zeros_l3208_320810


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l3208_320867

/-- The longest segment in a cylinder -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 12) :
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = Real.sqrt 244 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l3208_320867


namespace NUMINAMATH_CALUDE_solution_set_inequality_inequality_with_parameter_l3208_320856

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part I
theorem solution_set_inequality (x : ℝ) :
  (f x + f (x - 1) ≤ 2) ↔ (1/2 ≤ x ∧ x ≤ 5/2) :=
sorry

-- Theorem for part II
theorem inequality_with_parameter (a x : ℝ) (h : a > 0) :
  f (a * x) - a * f x ≤ f a :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_inequality_with_parameter_l3208_320856


namespace NUMINAMATH_CALUDE_first_day_earnings_10_l3208_320820

/-- A sequence of 5 numbers where each subsequent number is 4 more than the previous one -/
def IceCreamEarnings (first_day : ℕ) : Fin 5 → ℕ
  | ⟨0, _⟩ => first_day
  | ⟨n + 1, h⟩ => IceCreamEarnings first_day ⟨n, Nat.lt_trans n.lt_succ_self h⟩ + 4

/-- The theorem stating that if the sum of the sequence is 90, the first day's earnings were 10 -/
theorem first_day_earnings_10 :
  (∃ (first_day : ℕ), (Finset.sum Finset.univ (IceCreamEarnings first_day)) = 90) →
  (∃ (first_day : ℕ), (Finset.sum Finset.univ (IceCreamEarnings first_day)) = 90 ∧ first_day = 10) :=
by sorry


end NUMINAMATH_CALUDE_first_day_earnings_10_l3208_320820


namespace NUMINAMATH_CALUDE_counterfeit_coin_identification_l3208_320832

/-- Represents the outcome of a weighing on a balance scale -/
inductive WeighingResult
  | Equal : WeighingResult
  | Unequal : WeighingResult

/-- Represents a coin, which can be either real or counterfeit -/
inductive Coin
  | Real : Coin
  | Counterfeit : Coin

/-- Represents a weighing action on the balance scale -/
def weighing (c1 c2 : Coin) : WeighingResult :=
  match c1, c2 with
  | Coin.Real, Coin.Real => WeighingResult.Equal
  | Coin.Counterfeit, Coin.Real => WeighingResult.Unequal
  | Coin.Real, Coin.Counterfeit => WeighingResult.Unequal
  | Coin.Counterfeit, Coin.Counterfeit => WeighingResult.Equal

/-- Theorem stating that the counterfeit coin can be identified in at most 2 weighings -/
theorem counterfeit_coin_identification
  (coins : Fin 4 → Coin)
  (h_one_counterfeit : ∃! i, coins i = Coin.Counterfeit) :
  ∃ (w1 w2 : Fin 4 × Fin 4),
    let r1 := weighing (coins w1.1) (coins w1.2)
    let r2 := weighing (coins w2.1) (coins w2.2)
    ∃ i, coins i = Coin.Counterfeit ∧
         ∀ j, j ≠ i → coins j = Coin.Real :=
  sorry

end NUMINAMATH_CALUDE_counterfeit_coin_identification_l3208_320832


namespace NUMINAMATH_CALUDE_division_problem_l3208_320874

theorem division_problem (dividend : ℤ) (divisor : ℤ) (remainder : ℤ) (quotient : ℤ) :
  dividend = 12 →
  divisor = 17 →
  remainder = 8 →
  dividend = divisor * quotient + remainder →
  quotient = 0 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3208_320874


namespace NUMINAMATH_CALUDE_largest_fraction_equal_digit_sums_l3208_320880

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a number is four-digit -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- The largest fraction with equal digit sums -/
theorem largest_fraction_equal_digit_sums :
  ∀ m n : ℕ, 
    is_four_digit m → 
    is_four_digit n → 
    digit_sum m = digit_sum n → 
    (m : ℚ) / n ≤ 9900 / 1089 := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_equal_digit_sums_l3208_320880


namespace NUMINAMATH_CALUDE_pregnant_cows_l3208_320825

theorem pregnant_cows (total_cows : ℕ) (female_ratio : ℚ) (pregnant_ratio : ℚ) : 
  total_cows = 44 →
  female_ratio = 1/2 →
  pregnant_ratio = 1/2 →
  (↑total_cows * female_ratio * pregnant_ratio : ℚ) = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_pregnant_cows_l3208_320825


namespace NUMINAMATH_CALUDE_palindrome_product_sum_l3208_320842

/-- A function that checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- The theorem stating the sum of the two three-digit palindromes whose product is 522729 -/
theorem palindrome_product_sum : 
  ∃ (a b : ℕ), isThreeDigitPalindrome a ∧ 
                isThreeDigitPalindrome b ∧ 
                a * b = 522729 ∧ 
                a + b = 1366 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_product_sum_l3208_320842


namespace NUMINAMATH_CALUDE_jacket_price_change_l3208_320854

theorem jacket_price_change (P : ℝ) (x : ℝ) (h : x > 0) :
  P * (1 - (x / 100)^2) * 0.9 = 0.75 * P →
  x = 100 * Real.sqrt (1 / 6) := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_change_l3208_320854


namespace NUMINAMATH_CALUDE_xy_sum_square_l3208_320871

theorem xy_sum_square (x y : ℤ) 
  (h1 : x * y + x + y = 106) 
  (h2 : x^2 * y + x * y^2 = 1320) : 
  x^2 + y^2 = 748 ∨ x^2 + y^2 = 5716 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_square_l3208_320871


namespace NUMINAMATH_CALUDE_unique_number_property_l3208_320837

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 3 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l3208_320837


namespace NUMINAMATH_CALUDE_carmen_candle_usage_l3208_320896

/-- Calculates the number of candles used given the burning time per night and total nights -/
def candles_used (burn_time_per_night : ℚ) (total_nights : ℕ) : ℚ :=
  (burn_time_per_night * total_nights) / 8

theorem carmen_candle_usage :
  candles_used 2 24 = 6 := by sorry

end NUMINAMATH_CALUDE_carmen_candle_usage_l3208_320896


namespace NUMINAMATH_CALUDE_largest_x_floor_ratio_l3208_320883

theorem largest_x_floor_ratio : 
  ∀ x : ℝ, (↑(Int.floor x) / x = 8 / 9) → x ≤ 63 / 8 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_floor_ratio_l3208_320883


namespace NUMINAMATH_CALUDE_bread_weight_equals_antons_weight_l3208_320884

/-- Prove that the weight of bread eaten by Vladimir equals Anton's weight before his birthday -/
theorem bread_weight_equals_antons_weight 
  (A : ℝ) -- Anton's weight
  (B : ℝ) -- Vladimir's weight before eating bread
  (F : ℝ) -- Fyodor's weight
  (X : ℝ) -- Weight of the bread
  (h1 : X + F = A + B) -- Condition 1: Bread and Fyodor weigh as much as Anton and Vladimir
  (h2 : B + X = A + F) -- Condition 2: Vladimir's weight after eating equals Anton and Fyodor
  : X = A := by
  sorry

end NUMINAMATH_CALUDE_bread_weight_equals_antons_weight_l3208_320884


namespace NUMINAMATH_CALUDE_line_inclination_angle_l3208_320831

/-- The inclination angle of a line with equation x + √3 * y + c = 0 is 5π/6 --/
theorem line_inclination_angle (c : ℝ) : 
  let line := {(x, y) : ℝ × ℝ | x + Real.sqrt 3 * y + c = 0}
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < π ∧ 
    (∀ (x y : ℝ), (x, y) ∈ line → Real.tan θ = -(1 / Real.sqrt 3)) ∧
    θ = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l3208_320831


namespace NUMINAMATH_CALUDE_sheep_value_is_16_l3208_320818

/-- Represents the agreement between Kuba and the shepherd -/
structure Agreement where
  fullYearCoins : ℕ
  fullYearSheep : ℕ
  monthsWorked : ℕ
  coinsReceived : ℕ
  sheepReceived : ℕ

/-- Calculates the value of a sheep in gold coins based on the agreement -/
def sheepValue (a : Agreement) : ℕ :=
  let monthlyRate := a.fullYearCoins / 12
  let expectedCoins := monthlyRate * a.monthsWorked
  expectedCoins - a.coinsReceived

/-- The main theorem stating that the value of a sheep is 16 gold coins -/
theorem sheep_value_is_16 (a : Agreement) 
  (h1 : a.fullYearCoins = 20)
  (h2 : a.fullYearSheep = 1)
  (h3 : a.monthsWorked = 7)
  (h4 : a.coinsReceived = 5)
  (h5 : a.sheepReceived = 1) :
  sheepValue a = 16 := by
  sorry

#eval sheepValue { fullYearCoins := 20, fullYearSheep := 1, monthsWorked := 7, coinsReceived := 5, sheepReceived := 1 }

end NUMINAMATH_CALUDE_sheep_value_is_16_l3208_320818


namespace NUMINAMATH_CALUDE_a_5_value_l3208_320846

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem a_5_value (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 * a 7 = 2 →
  a 3 + a 7 = -4 →
  a 5 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_a_5_value_l3208_320846


namespace NUMINAMATH_CALUDE_pencil_purchase_count_l3208_320885

/-- Represents the number of pencils and pens purchased -/
structure Purchase where
  pencils : ℕ
  pens : ℕ

/-- Represents the cost in won -/
@[reducible] def Won := ℕ

theorem pencil_purchase_count (p : Purchase) 
  (h1 : p.pencils + p.pens = 12)
  (h2 : 1000 * p.pencils + 1300 * p.pens = 15000) :
  p.pencils = 2 := by
  sorry

#check pencil_purchase_count

end NUMINAMATH_CALUDE_pencil_purchase_count_l3208_320885


namespace NUMINAMATH_CALUDE_mr_grey_purchases_l3208_320894

/-- The cost of Mr. Grey's purchases -/
theorem mr_grey_purchases (polo_price : ℝ) : polo_price = 26 :=
  let necklace_price := 83
  let game_price := 90
  let rebate := 12
  let total_cost := 322
  let num_polos := 3
  let num_necklaces := 2
  have h : num_polos * polo_price + num_necklaces * necklace_price + game_price - rebate = total_cost :=
    by sorry
  sorry

end NUMINAMATH_CALUDE_mr_grey_purchases_l3208_320894


namespace NUMINAMATH_CALUDE_timothys_journey_speed_l3208_320888

/-- Proves that given the conditions of Timothy's journey, his average speed during the first part was 10 mph. -/
theorem timothys_journey_speed (v : ℝ) (T : ℝ) (h1 : T > 0) :
  v * (0.25 * T) + 50 * (0.75 * T) = 40 * T →
  v = 10 := by
  sorry

end NUMINAMATH_CALUDE_timothys_journey_speed_l3208_320888


namespace NUMINAMATH_CALUDE_farm_animals_l3208_320893

theorem farm_animals (cows chickens : ℕ) : 
  cows + chickens = 12 →
  4 * cows + 2 * chickens = 20 + 2 * (cows + chickens) →
  cows = 10 := by sorry

end NUMINAMATH_CALUDE_farm_animals_l3208_320893


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3208_320828

/-- Given a circle with equation x^2 + y^2 - 2x + 6y + 6 = 0, prove that its center is at (1, -3) and its radius is 2 -/
theorem circle_center_and_radius :
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 - 2*x + 6*y + 6 = 0
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -3) ∧ 
    radius = 2 ∧
    ∀ (x y : ℝ), circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3208_320828


namespace NUMINAMATH_CALUDE_expression_evaluation_l3208_320881

theorem expression_evaluation : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3208_320881


namespace NUMINAMATH_CALUDE_four_inch_gold_cube_value_l3208_320877

/-- The value of a cube of gold given its side length -/
def gold_value (side : ℕ) : ℚ :=
  let base_value : ℚ := 300
  let base_side : ℕ := 1
  let increase_rate : ℚ := 1.1
  let value_per_cubic_inch : ℚ := base_value * (increase_rate ^ (side - base_side))
  (side ^ 3 : ℚ) * value_per_cubic_inch

/-- Theorem stating the value of a 4-inch cube of gold -/
theorem four_inch_gold_cube_value :
  ⌊gold_value 4⌋ = 25555 :=
sorry

end NUMINAMATH_CALUDE_four_inch_gold_cube_value_l3208_320877


namespace NUMINAMATH_CALUDE_revenue_change_l3208_320839

theorem revenue_change (R : ℝ) (x : ℝ) (h : R > 0) :
  R * (1 + x / 100) * (1 - x / 100) = R * 0.96 →
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l3208_320839


namespace NUMINAMATH_CALUDE_system_solutions_l3208_320845

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  x^3 + y^3 = 3*y + 3*z + 4 ∧
  y^3 + z^3 = 3*z + 3*x + 4 ∧
  z^3 + x^3 = 3*x + 3*y + 4

/-- The solutions to the system of equations -/
theorem system_solutions :
  (∀ x y z : ℝ, system x y z ↔ (x = -1 ∧ y = -1 ∧ z = -1) ∨ (x = 2 ∧ y = 2 ∧ z = 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l3208_320845


namespace NUMINAMATH_CALUDE_division_problem_solution_l3208_320835

theorem division_problem_solution :
  ∀ (D d q r : ℕ),
    D + d + q + r = 205 →
    q = d →
    D = d * q + r →
    D = 174 ∧ d = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_division_problem_solution_l3208_320835


namespace NUMINAMATH_CALUDE_matchstick_houses_l3208_320897

theorem matchstick_houses (initial_matchsticks : ℕ) (num_houses : ℕ) 
  (h1 : initial_matchsticks = 600)
  (h2 : num_houses = 30) :
  (initial_matchsticks / 2) / num_houses = 10 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_houses_l3208_320897


namespace NUMINAMATH_CALUDE_unique_division_problem_l3208_320892

theorem unique_division_problem :
  ∀ (a b : ℕ),
  (a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 1 ∧ b ≤ 9) →
  (∃ (p : ℕ), 111111 * a = 1111 * b * 233 + p) →
  (∃ (q : ℕ), 11111 * a = 111 * b * 233 + (q - 1000)) →
  (a = 7 ∧ b = 3) := by
sorry

end NUMINAMATH_CALUDE_unique_division_problem_l3208_320892


namespace NUMINAMATH_CALUDE_representable_integers_l3208_320830

/-- Represents an arithmetic expression using only the digit 2 and basic operations -/
inductive Expr
  | two : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluates an arithmetic expression -/
def eval : Expr → ℚ
  | Expr.two => 2
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Counts the number of 2's used in an expression -/
def count_twos : Expr → ℕ
  | Expr.two => 1
  | Expr.add e1 e2 => count_twos e1 + count_twos e2
  | Expr.sub e1 e2 => count_twos e1 + count_twos e2
  | Expr.mul e1 e2 => count_twos e1 + count_twos e2
  | Expr.div e1 e2 => count_twos e1 + count_twos e2

theorem representable_integers :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2019 →
  ∃ e : Expr, eval e = n ∧ count_twos e ≤ 17 :=
sorry

end NUMINAMATH_CALUDE_representable_integers_l3208_320830


namespace NUMINAMATH_CALUDE_triangle_square_side_ratio_l3208_320849

theorem triangle_square_side_ratio :
  ∀ (t s : ℝ),
  (3 * t = 15) →  -- Perimeter of equilateral triangle
  (4 * s = 12) →  -- Perimeter of square
  (t / s = 5 / 3) :=  -- Ratio of side lengths
by
  sorry

end NUMINAMATH_CALUDE_triangle_square_side_ratio_l3208_320849


namespace NUMINAMATH_CALUDE_driveway_snow_volume_l3208_320809

/-- Calculates the total volume of snow on a driveway with given dimensions and snow depths -/
theorem driveway_snow_volume 
  (driveway_length : ℝ) 
  (driveway_width : ℝ) 
  (section1_length : ℝ) 
  (section1_depth : ℝ) 
  (section2_length : ℝ) 
  (section2_depth : ℝ) 
  (h1 : driveway_length = 30) 
  (h2 : driveway_width = 3) 
  (h3 : section1_length = 10) 
  (h4 : section1_depth = 1) 
  (h5 : section2_length = 20) 
  (h6 : section2_depth = 0.5) 
  (h7 : section1_length + section2_length = driveway_length) : 
  section1_length * driveway_width * section1_depth + 
  section2_length * driveway_width * section2_depth = 60 :=
by
  sorry

#check driveway_snow_volume

end NUMINAMATH_CALUDE_driveway_snow_volume_l3208_320809


namespace NUMINAMATH_CALUDE_profit_calculation_l3208_320821

/-- Represents the profit made from commercial farming -/
def profit_from_farming 
  (total_land : ℝ)           -- Total land area in hectares
  (num_sons : ℕ)             -- Number of sons
  (profit_per_son : ℝ)       -- Annual profit per son
  (land_unit : ℝ)            -- Land unit for profit calculation in m^2
  : ℝ :=
  -- The function body is left empty as we only need the statement
  sorry

/-- Theorem stating the profit from farming under given conditions -/
theorem profit_calculation :
  let total_land : ℝ := 3                    -- 3 hectares
  let num_sons : ℕ := 8                      -- 8 sons
  let profit_per_son : ℝ := 10000            -- $10,000 per year per son
  let land_unit : ℝ := 750                   -- 750 m^2 unit
  let hectare_to_sqm : ℝ := 10000            -- 1 hectare = 10,000 m^2
  profit_from_farming total_land num_sons profit_per_son land_unit = 500 :=
by
  sorry  -- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_profit_calculation_l3208_320821


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3208_320887

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ 
    ((x = -7 ∧ y = -99) ∨ 
     (x = -1 ∧ y = -9) ∨ 
     (x = 1 ∧ y = 5) ∨ 
     (x = 7 ∧ y = -97)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3208_320887


namespace NUMINAMATH_CALUDE_closest_multiple_of_15_to_2023_l3208_320805

theorem closest_multiple_of_15_to_2023 :
  ∀ k : ℤ, |15 * k - 2023| ≥ |2025 - 2023| := by
  sorry

end NUMINAMATH_CALUDE_closest_multiple_of_15_to_2023_l3208_320805


namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l3208_320829

theorem sum_of_roots_zero (p q : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ x = p ∨ x = q) → 
  p = 2*q → 
  p + q = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_zero_l3208_320829


namespace NUMINAMATH_CALUDE_profit_calculation_l3208_320866

-- Define package prices
def basic_price : ℕ := 5
def deluxe_price : ℕ := 10
def premium_price : ℕ := 15

-- Define weekday car wash numbers
def basic_cars : ℕ := 50
def deluxe_cars : ℕ := 40
def premium_cars : ℕ := 20

-- Define employee wages
def employee_a_wage : ℕ := 110
def employee_b_wage : ℕ := 90
def employee_c_wage : ℕ := 100
def employee_d_wage : ℕ := 80

-- Define operating expenses
def weekday_expenses : ℕ := 200

-- Define the number of weekdays
def weekdays : ℕ := 5

-- Define the function to calculate total profit
def total_profit : ℕ :=
  let daily_revenue := basic_price * basic_cars + deluxe_price * deluxe_cars + premium_price * premium_cars
  let total_revenue := daily_revenue * weekdays
  let employee_expenses := employee_a_wage * 5 + employee_b_wage * 2 + employee_c_wage * 3 + employee_d_wage * 2
  let total_expenses := employee_expenses + weekday_expenses * weekdays
  total_revenue - total_expenses

-- Theorem statement
theorem profit_calculation : total_profit = 2560 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l3208_320866


namespace NUMINAMATH_CALUDE_constant_product_equals_one_fourth_l3208_320811

/-- Given a function f(x) = (bx + 1) / (2x + a), where a and b are constants,
    and ab ≠ 2, prove that if f(x) * f(1/x) is constant for all x ≠ 0,
    then this constant equals 1/4. -/
theorem constant_product_equals_one_fourth
  (a b : ℝ) (h : a * b ≠ 2)
  (f : ℝ → ℝ)
  (hf : ∀ x, x ≠ 0 → f x = (b * x + 1) / (2 * x + a))
  (h_constant : ∃ k, ∀ x, x ≠ 0 → f x * f (1/x) = k) :
  ∃ k, (∀ x, x ≠ 0 → f x * f (1/x) = k) ∧ k = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_constant_product_equals_one_fourth_l3208_320811


namespace NUMINAMATH_CALUDE_sqrt_square_not_always_equal_l3208_320836

theorem sqrt_square_not_always_equal (a : ℝ) : ¬(∀ a, Real.sqrt (a^2) = a) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_not_always_equal_l3208_320836


namespace NUMINAMATH_CALUDE_unique_solution_l3208_320868

/-- A three-digit number represented as a tuple of its digits -/
def ThreeDigitNumber := (ℕ × ℕ × ℕ)

/-- Convert a ThreeDigitNumber to its integer representation -/
def to_int (n : ThreeDigitNumber) : ℕ :=
  100 * n.1 + 10 * n.2.1 + n.2.2

/-- Check if a ThreeDigitNumber satisfies the condition abc = (a + b + c)^3 -/
def satisfies_condition (n : ThreeDigitNumber) : Prop :=
  to_int n = (n.1 + n.2.1 + n.2.2) ^ 3

/-- The theorem stating that 512 is the only solution -/
theorem unique_solution :
  ∃! (n : ThreeDigitNumber), 
    100 ≤ to_int n ∧ 
    to_int n ≤ 999 ∧ 
    satisfies_condition n ∧
    to_int n = 512 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3208_320868


namespace NUMINAMATH_CALUDE_coat_price_theorem_l3208_320824

theorem coat_price_theorem (price : ℝ) : 
  (price - 150 = price * (1 - 0.3)) → price = 500 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_theorem_l3208_320824


namespace NUMINAMATH_CALUDE_baker_production_theorem_l3208_320875

/-- Represents the baker's bread production over a period of time. -/
structure BakerProduction where
  loaves_per_oven_hour : ℕ
  num_ovens : ℕ
  weekday_hours : ℕ
  weekend_hours : ℕ
  num_weeks : ℕ

/-- Calculates the total number of loaves baked over the given period. -/
def total_loaves (bp : BakerProduction) : ℕ :=
  let loaves_per_hour := bp.loaves_per_oven_hour * bp.num_ovens
  let weekday_loaves := loaves_per_hour * bp.weekday_hours * 5
  let weekend_loaves := loaves_per_hour * bp.weekend_hours * 2
  (weekday_loaves + weekend_loaves) * bp.num_weeks

/-- Theorem stating that given the baker's production conditions, 
    the total number of loaves baked in 3 weeks is 1740. -/
theorem baker_production_theorem (bp : BakerProduction) 
  (h1 : bp.loaves_per_oven_hour = 5)
  (h2 : bp.num_ovens = 4)
  (h3 : bp.weekday_hours = 5)
  (h4 : bp.weekend_hours = 2)
  (h5 : bp.num_weeks = 3) :
  total_loaves bp = 1740 := by
  sorry

#eval total_loaves ⟨5, 4, 5, 2, 3⟩

end NUMINAMATH_CALUDE_baker_production_theorem_l3208_320875


namespace NUMINAMATH_CALUDE_barbell_to_rack_ratio_is_one_to_ten_l3208_320865

/-- Given a squat rack cost and total cost, calculates the ratio of barbell cost to squat rack cost -/
def barbellToRackRatio (rackCost totalCost : ℚ) : ℚ × ℚ :=
  let barbellCost := totalCost - rackCost
  (barbellCost, rackCost)

/-- Theorem: The ratio of barbell cost to squat rack cost is 1:10 for given costs -/
theorem barbell_to_rack_ratio_is_one_to_ten :
  barbellToRackRatio 2500 2750 = (1, 10) := by
  sorry

#eval barbellToRackRatio 2500 2750

end NUMINAMATH_CALUDE_barbell_to_rack_ratio_is_one_to_ten_l3208_320865


namespace NUMINAMATH_CALUDE_feet_heads_difference_l3208_320857

theorem feet_heads_difference : 
  let birds : ℕ := 4
  let dogs : ℕ := 3
  let cats : ℕ := 18
  let humans : ℕ := 7
  let bird_feet : ℕ := 2
  let dog_feet : ℕ := 4
  let cat_feet : ℕ := 4
  let human_feet : ℕ := 2
  let total_heads : ℕ := birds + dogs + cats + humans
  let total_feet : ℕ := birds * bird_feet + dogs * dog_feet + cats * cat_feet + humans * human_feet
  total_feet - total_heads = 74 :=
by sorry

end NUMINAMATH_CALUDE_feet_heads_difference_l3208_320857


namespace NUMINAMATH_CALUDE_line_through_points_specific_line_equation_l3208_320886

/-- A line passing through two given points has a specific equation -/
theorem line_through_points (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂) :
  ∃ k b : ℝ, ∀ x y : ℝ, y = k * x + b ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) :=
sorry

/-- The line passing through (2, 5) and (1, 1) has the equation y = 4x - 3 -/
theorem specific_line_equation :
  ∃ k b : ℝ, (k = 4 ∧ b = -3) ∧
    (∀ x y : ℝ, y = k * x + b ↔ (x = 2 ∧ y = 5) ∨ (x = 1 ∧ y = 1)) :=
sorry

end NUMINAMATH_CALUDE_line_through_points_specific_line_equation_l3208_320886


namespace NUMINAMATH_CALUDE_logical_reasoning_methods_correct_answer_is_C_l3208_320840

-- Define the reasoning methods
inductive ReasoningMethod
| SphereFromCircle
| TriangleAngleSum
| ClassPerformance
| PolygonAngleSum

-- Define a predicate for logical reasoning
def isLogical : ReasoningMethod → Prop
| ReasoningMethod.SphereFromCircle => True
| ReasoningMethod.TriangleAngleSum => True
| ReasoningMethod.ClassPerformance => False
| ReasoningMethod.PolygonAngleSum => True

-- Theorem stating which reasoning methods are logical
theorem logical_reasoning_methods :
  (isLogical ReasoningMethod.SphereFromCircle) ∧
  (isLogical ReasoningMethod.TriangleAngleSum) ∧
  (¬isLogical ReasoningMethod.ClassPerformance) ∧
  (isLogical ReasoningMethod.PolygonAngleSum) :=
by sorry

-- Define the answer options
inductive AnswerOption
| A
| B
| C
| D

-- Define the correct answer
def correctAnswer : AnswerOption := AnswerOption.C

-- Theorem stating that C is the correct answer
theorem correct_answer_is_C :
  correctAnswer = AnswerOption.C :=
by sorry

end NUMINAMATH_CALUDE_logical_reasoning_methods_correct_answer_is_C_l3208_320840


namespace NUMINAMATH_CALUDE_arithmetic_progression_equality_l3208_320851

theorem arithmetic_progression_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((a + b) / 2 = (Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2)) / 2) → a = b := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_equality_l3208_320851


namespace NUMINAMATH_CALUDE_complement_intersection_equals_l3208_320852

def U : Set ℕ := {1, 2, 3, 4}
def P : Set ℕ := {2, 3}
def Q : Set ℕ := {3, 4}

theorem complement_intersection_equals :
  (U \ (P ∩ Q)) = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_l3208_320852


namespace NUMINAMATH_CALUDE_food_price_calculation_l3208_320800

/-- The original food price before tax and tip -/
def original_price : ℝ := 160

/-- The sales tax rate -/
def sales_tax_rate : ℝ := 0.1

/-- The tip rate -/
def tip_rate : ℝ := 0.2

/-- The total bill amount -/
def total_bill : ℝ := 211.20

theorem food_price_calculation :
  original_price * (1 + sales_tax_rate) * (1 + tip_rate) = total_bill := by
  sorry


end NUMINAMATH_CALUDE_food_price_calculation_l3208_320800


namespace NUMINAMATH_CALUDE_min_value_theorem_l3208_320864

theorem min_value_theorem (c a b : ℝ) (hc : c > 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (heq : 4 * a^2 - 2 * a * b + b^2 - c = 0)
  (hmax : ∀ (a' b' : ℝ), a' ≠ 0 → b' ≠ 0 → 4 * a'^2 - 2 * a' * b' + b'^2 - c = 0 →
    |2 * a + b| ≥ |2 * a' + b'|) :
  (1 / a + 2 / b + 4 / c) ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3208_320864


namespace NUMINAMATH_CALUDE_pears_given_by_mike_l3208_320813

theorem pears_given_by_mike (initial_pears : ℕ) (pears_given_away : ℕ) (final_pears : ℕ) :
  initial_pears = 46 →
  pears_given_away = 47 →
  final_pears = 11 →
  pears_given_away - initial_pears + final_pears = 12 :=
by sorry

end NUMINAMATH_CALUDE_pears_given_by_mike_l3208_320813


namespace NUMINAMATH_CALUDE_xyz_value_l3208_320882

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 24) :
  x * y * z = 4 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l3208_320882
