import Mathlib

namespace carnival_game_earnings_l530_53017

/-- The daily earnings of a carnival game -/
def daily_earnings (total_earnings : ℕ) (num_days : ℕ) : ℚ :=
  total_earnings / num_days

/-- Theorem stating that the daily earnings are $144 -/
theorem carnival_game_earnings : daily_earnings 3168 22 = 144 := by
  sorry

end carnival_game_earnings_l530_53017


namespace negation_of_universal_proposition_l530_53046

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 2 → x^2 > 4) ↔ (∃ x : ℝ, x > 2 ∧ x^2 ≤ 4) := by
  sorry

end negation_of_universal_proposition_l530_53046


namespace max_value_sqrt_sum_max_value_achieved_l530_53044

theorem max_value_sqrt_sum (x : ℝ) (h : x ∈ Set.Icc (-49) 49) : 
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 :=
by sorry

theorem max_value_achieved : 
  ∃ x, x ∈ Set.Icc (-49) 49 ∧ Real.sqrt (49 + x) + Real.sqrt (49 - x) = 14 :=
by sorry

end max_value_sqrt_sum_max_value_achieved_l530_53044


namespace money_distribution_l530_53014

/-- Given three people A, B, and C with the following money distribution:
    - A, B, and C have Rs. 700 in total
    - A and C together have Rs. 300
    - C has Rs. 200
    Prove that B and C together have Rs. 600 -/
theorem money_distribution (total : ℕ) (ac_sum : ℕ) (c_money : ℕ) :
  total = 700 →
  ac_sum = 300 →
  c_money = 200 →
  ∃ (a b : ℕ), a + b + c_money = total ∧ a + c_money = ac_sum ∧ b + c_money = 600 :=
by sorry

end money_distribution_l530_53014


namespace bowling_ball_weight_proof_l530_53021

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 17

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 34

theorem bowling_ball_weight_proof :
  (10 * bowling_ball_weight = 5 * canoe_weight) ∧
  (3 * canoe_weight = 102) →
  bowling_ball_weight = 17 := by
  sorry

end bowling_ball_weight_proof_l530_53021


namespace power_equation_solution_l530_53054

theorem power_equation_solution : ∃ x : ℕ, 125^2 = 5^x ∧ x = 6 := by sorry

end power_equation_solution_l530_53054


namespace rationalize_sqrt_five_twelfths_l530_53098

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end rationalize_sqrt_five_twelfths_l530_53098


namespace adams_shelves_l530_53088

theorem adams_shelves (figures_per_shelf : ℕ) (total_figures : ℕ) (h1 : figures_per_shelf = 10) (h2 : total_figures = 80) :
  total_figures / figures_per_shelf = 8 := by
sorry

end adams_shelves_l530_53088


namespace expression_simplification_l530_53023

theorem expression_simplification (a b : ℝ) (h1 : 0 < a) (h2 : a < 2*b) :
  1.15 * (Real.sqrt (a^2 - 4*a*b + 4*b^2) / Real.sqrt (a^2 + 4*a*b + 4*b^2)) - 
  (8*a*b / (a^2 - 4*b^2)) + (2*b / (a - 2*b)) = a / (2*b - a) := by
  sorry

end expression_simplification_l530_53023


namespace geometric_sequence_property_l530_53084

/-- A geometric sequence with all positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_roots : ∃ m : ℝ, 2 * (a 5)^2 - m * (a 5) + 2 * Real.exp 4 = 0 ∧
                      2 * (a 13)^2 - m * (a 13) + 2 * Real.exp 4 = 0) :
  a 7 * a 9 * a 11 = Real.exp 6 := by
sorry

end geometric_sequence_property_l530_53084


namespace division_remainder_problem_l530_53043

theorem division_remainder_problem (a b r : ℕ) : 
  a - b = 2500 → 
  a = 2982 → 
  ∃ q, a = q * b + r ∧ q = 6 ∧ r < b → 
  r = 90 := by
sorry

end division_remainder_problem_l530_53043


namespace cubic_roots_sum_l530_53061

theorem cubic_roots_sum (p q r : ℝ) : 
  (3 * p^3 - 4 * p^2 + 200 * p - 5 = 0) →
  (3 * q^3 - 4 * q^2 + 200 * q - 5 = 0) →
  (3 * r^3 - 4 * r^2 + 200 * r - 5 = 0) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 403 := by
sorry

end cubic_roots_sum_l530_53061


namespace shaded_area_of_rotated_diameters_l530_53093

theorem shaded_area_of_rotated_diameters (r : ℝ) (h : r = 6) :
  let circle_area := π * r^2
  let quadrant_area := circle_area / 4
  let triangle_area := r^2
  2 * quadrant_area + 2 * triangle_area = 72 + 9 * π := by
  sorry

end shaded_area_of_rotated_diameters_l530_53093


namespace boxes_given_to_mother_l530_53022

theorem boxes_given_to_mother (initial_boxes : ℕ) (final_boxes : ℕ) : 
  initial_boxes = 9 →
  final_boxes = 4 →
  final_boxes * 2 = initial_boxes - (initial_boxes - final_boxes * 2) :=
by
  sorry

end boxes_given_to_mother_l530_53022


namespace inequality_proof_l530_53089

theorem inequality_proof (a b : ℝ) (n : ℕ) (h : 2 * a + 3 * b = 12) :
  (a / 3) ^ n + (b / 2) ^ n ≥ 2 := by
  sorry

end inequality_proof_l530_53089


namespace parallel_line_through_point_A_l530_53063

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Define the point A
def point_A : ℝ × ℝ := (-1, 0)

-- Define the parallel line passing through point A
def parallel_line (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Theorem statement
theorem parallel_line_through_point_A :
  (∀ x y : ℝ, given_line x y ↔ 2 * x - y + 1 = 0) →
  parallel_line point_A.1 point_A.2 ∧
  ∀ x y : ℝ, parallel_line x y → 
    ∃ k : ℝ, y - point_A.2 = k * (x - point_A.1) ∧
             2 = k * 1 :=
by sorry

end parallel_line_through_point_A_l530_53063


namespace robin_phone_pictures_l530_53070

theorem robin_phone_pictures (phone_pics camera_pics albums pics_per_album : ℕ) :
  camera_pics = 5 →
  albums = 5 →
  pics_per_album = 8 →
  phone_pics + camera_pics = albums * pics_per_album →
  phone_pics = 35 := by
  sorry

end robin_phone_pictures_l530_53070


namespace company_max_people_l530_53067

/-- Represents a company with three clubs and their membership information -/
structure Company where
  M : ℕ  -- Number of people in club M
  S : ℕ  -- Number of people in club S
  Z : ℕ  -- Number of people in club Z
  none : ℕ  -- Maximum number of people not in any club

/-- The maximum number of people in the company -/
def Company.maxPeople (c : Company) : ℕ := c.M + c.S + c.Z + c.none

/-- Theorem stating the maximum number of people in the company under given conditions -/
theorem company_max_people :
  ∀ (c : Company),
  c.M = 16 →
  c.S = 18 →
  c.Z = 11 →
  c.none ≤ 26 →
  c.maxPeople ≤ 71 := by
  sorry

#check company_max_people

end company_max_people_l530_53067


namespace maggie_goldfish_fraction_l530_53087

def total_goldfish : ℕ := 100
def allowed_fraction : ℚ := 1 / 2
def remaining_to_catch : ℕ := 20

theorem maggie_goldfish_fraction :
  let allowed_total := total_goldfish * allowed_fraction
  let caught := allowed_total - remaining_to_catch
  caught / allowed_total = 3 / 5 := by
sorry

end maggie_goldfish_fraction_l530_53087


namespace total_sharks_count_l530_53086

/-- The number of sharks at Newport Beach -/
def newport_sharks : ℕ := 22

/-- The number of sharks at Dana Point beach -/
def dana_point_sharks : ℕ := 4 * newport_sharks

/-- The total number of sharks on both beaches -/
def total_sharks : ℕ := newport_sharks + dana_point_sharks

theorem total_sharks_count : total_sharks = 110 := by
  sorry

end total_sharks_count_l530_53086


namespace square_perimeter_with_circles_l530_53082

/-- Represents a square with inscribed circles -/
structure SquareWithCircles where
  /-- Side length of the square -/
  side : ℝ
  /-- Radius of each inscribed circle -/
  circle_radius : ℝ
  /-- The circles touch two sides of the square and two respective corners -/
  circles_touch_sides : circle_radius > 0

/-- The perimeter of a square with inscribed circles of radius 4 is 64√2 -/
theorem square_perimeter_with_circles (s : SquareWithCircles) 
  (h : s.circle_radius = 4) : s.side * 4 = 64 * Real.sqrt 2 := by
  sorry

#check square_perimeter_with_circles

end square_perimeter_with_circles_l530_53082


namespace baseball_card_value_decrease_l530_53024

theorem baseball_card_value_decrease (x : ℝ) : 
  (1 - x / 100) * 0.9 = 0.36 → x = 60 :=
by
  sorry

end baseball_card_value_decrease_l530_53024


namespace series_sum_equals_four_l530_53029

/-- The sum of the series ∑(4n+2)/3^n from n=1 to infinity equals 4. -/
theorem series_sum_equals_four :
  (∑' n : ℕ, (4 * n + 2) / (3 : ℝ) ^ n) = 4 := by
  sorry

end series_sum_equals_four_l530_53029


namespace largest_x_sqrt_3x_eq_5x_l530_53058

theorem largest_x_sqrt_3x_eq_5x : 
  ∃ (x_max : ℚ), x_max = 3/25 ∧ 
  (∀ x : ℚ, x ≥ 0 → Real.sqrt (3*x) = 5*x → x ≤ x_max) ∧
  Real.sqrt (3*x_max) = 5*x_max := by
sorry

end largest_x_sqrt_3x_eq_5x_l530_53058


namespace holly_blood_pressure_pills_l530_53073

/-- Represents the number of pills Holly takes daily for each medication type -/
structure DailyPills where
  insulin : ℕ
  bloodPressure : ℕ
  anticonvulsant : ℕ

/-- Calculates the total number of pills taken in a week -/
def weeklyTotal (d : DailyPills) : ℕ :=
  7 * (d.insulin + d.bloodPressure + d.anticonvulsant)

theorem holly_blood_pressure_pills
  (d : DailyPills)
  (h1 : d.insulin = 2)
  (h2 : d.anticonvulsant = 2 * d.bloodPressure)
  (h3 : weeklyTotal d = 77) :
  d.bloodPressure = 3 := by
sorry

end holly_blood_pressure_pills_l530_53073


namespace rectangle_min_area_l530_53051

/-- A rectangle with integer dimensions and perimeter 60 has minimum area 29 when the shorter dimension is minimized. -/
theorem rectangle_min_area : ∀ l w : ℕ,
  l > 0 → w > 0 →
  2 * (l + w) = 60 →
  ∀ l' w' : ℕ, l' > 0 → w' > 0 → 2 * (l' + w') = 60 →
  min l w ≤ min l' w' →
  l * w ≥ 29 := by
sorry

end rectangle_min_area_l530_53051


namespace sara_lunch_cost_theorem_l530_53079

/-- The cost of Sara's lunch given the prices of a hotdog and a salad -/
def lunch_cost (hotdog_price salad_price : ℚ) : ℚ :=
  hotdog_price + salad_price

/-- Theorem stating that Sara's lunch cost is the sum of hotdog and salad prices -/
theorem sara_lunch_cost_theorem (hotdog_price salad_price : ℚ) :
  lunch_cost hotdog_price salad_price = hotdog_price + salad_price :=
by sorry

end sara_lunch_cost_theorem_l530_53079


namespace rightmost_bag_balls_l530_53038

/-- The number of bags -/
def n : ℕ := 2003

/-- The number of balls in every 7 consecutive bags -/
def total_balls_in_seven : ℕ := 19

/-- The period of the ball distribution -/
def period : ℕ := 7

/-- The number of balls in the leftmost bag -/
def R : ℕ := total_balls_in_seven - (period - 1)

/-- A function representing the number of balls in the i-th bag -/
def balls (i : ℕ) : ℕ :=
  if i % period = 1 then R else (total_balls_in_seven - R) / (period - 1)

/-- The theorem to be proved -/
theorem rightmost_bag_balls : balls n = 8 := by
  sorry

end rightmost_bag_balls_l530_53038


namespace rhombus_perimeter_l530_53015

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end rhombus_perimeter_l530_53015


namespace pizza_promotion_savings_l530_53045

/-- Represents the pizza promotion savings calculation --/
theorem pizza_promotion_savings : 
  let regular_medium_price : ℚ := 18
  let discounted_medium_price : ℚ := 5
  let num_medium_pizzas : ℕ := 3
  let large_pizza_toppings_cost : ℚ := 2 + 1.5 + 1 + 2.5
  let medium_pizza_toppings_cost : ℚ := large_pizza_toppings_cost

  let medium_pizza_savings : ℚ := (regular_medium_price - discounted_medium_price) * num_medium_pizzas
  let toppings_savings : ℚ := medium_pizza_toppings_cost * num_medium_pizzas

  let total_savings : ℚ := medium_pizza_savings + toppings_savings

  total_savings = 60 := by
  sorry

end pizza_promotion_savings_l530_53045


namespace opposite_sides_range_l530_53069

/-- Two points are on opposite sides of a line if and only if the product of their distances from the line is negative. -/
def opposite_sides (x₁ y₁ x₂ y₂ a b c : ℝ) : Prop :=
  (a * x₁ + b * y₁ + c) * (a * x₂ + b * y₂ + c) < 0

/-- The theorem stating the range of m for which (1, 2) and (1, 1) are on opposite sides of 3x - y + m = 0. -/
theorem opposite_sides_range :
  ∀ m : ℝ, opposite_sides 1 2 1 1 3 (-1) m ↔ -2 < m ∧ m < -1 := by
  sorry

end opposite_sides_range_l530_53069


namespace max_ratio_hyperbola_areas_max_ratio_hyperbola_areas_equality_l530_53019

theorem max_ratio_hyperbola_areas (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a * b) / (a^2 + b^2) ≤ (1 : ℝ) / 2 :=
sorry

theorem max_ratio_hyperbola_areas_equality (a : ℝ) (ha : a > 0) : 
  (a * a) / (a^2 + a^2) = (1 : ℝ) / 2 :=
sorry

end max_ratio_hyperbola_areas_max_ratio_hyperbola_areas_equality_l530_53019


namespace smallest_three_digit_number_satisfying_conditions_l530_53001

theorem smallest_three_digit_number_satisfying_conditions : ∃ n : ℕ,
  (100 ≤ n ∧ n ≤ 999) ∧
  (∃ k : ℕ, n + 7 = 9 * k) ∧
  (∃ m : ℕ, n - 6 = 7 * m) ∧
  (∀ x : ℕ, (100 ≤ x ∧ x < n) → ¬((∃ k : ℕ, x + 7 = 9 * k) ∧ (∃ m : ℕ, x - 6 = 7 * m))) ∧
  n = 116 :=
by sorry

end smallest_three_digit_number_satisfying_conditions_l530_53001


namespace sum_of_triangles_is_26_l530_53075

-- Define the triangle operation
def triangleOp (a b c : ℚ) : ℚ := a * b / c

-- Define the sum of two triangle operations
def sumTriangleOps (a1 b1 c1 a2 b2 c2 : ℚ) : ℚ :=
  triangleOp a1 b1 c1 + triangleOp a2 b2 c2

-- Theorem statement
theorem sum_of_triangles_is_26 :
  sumTriangleOps 4 8 2 5 10 5 = 26 := by
  sorry

end sum_of_triangles_is_26_l530_53075


namespace trajectory_of_M_line_l_equation_when_OP_eq_OM_area_POM_when_OP_eq_OM_l530_53094

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*y = 0

-- Define point P
def point_P : ℝ × ℝ := (2, 2)

-- Define the line l passing through P and intersecting C
def line_l (m : ℝ) (x y : ℝ) : Prop := y - 2 = m * (x - 2)

-- Define the midpoint M of AB
def point_M (x y : ℝ) : Prop := ∃ (x_a y_a x_b y_b : ℝ),
  circle_C x_a y_a ∧ circle_C x_b y_b ∧
  line_l ((y_b - y_a) / (x_b - x_a)) x_a y_a ∧
  line_l ((y_b - y_a) / (x_b - x_a)) x_b y_b ∧
  x = (x_a + x_b) / 2 ∧ y = (y_a + y_b) / 2

-- Theorem 1: Trajectory of M
theorem trajectory_of_M :
  ∀ (x y : ℝ), point_M x y → (x - 1)^2 + (y - 3)^2 = 2 :=
sorry

-- Theorem 2a: Equation of line l when |OP| = |OM|
theorem line_l_equation_when_OP_eq_OM :
  ∀ (x y : ℝ), point_M x y → (x^2 + y^2 = x^2 + (y - 3)^2 + 5) →
  (x + 3*y - 8 = 0) :=
sorry

-- Theorem 2b: Area of triangle POM when |OP| = |OM|
theorem area_POM_when_OP_eq_OM :
  ∀ (x y : ℝ), point_M x y → (x^2 + y^2 = x^2 + (y - 3)^2 + 5) →
  (1/2 * |x - 2| * |y - 2| = 16/5) :=
sorry

end trajectory_of_M_line_l_equation_when_OP_eq_OM_area_POM_when_OP_eq_OM_l530_53094


namespace line_slope_intercept_sum_l530_53008

/-- Given a line passing through points (1, -2) and (3, 4), 
    prove that the sum of its slope and y-intercept is -2 -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℝ), 
  (∀ (x y : ℝ), y = m * x + b → 
    ((x = 1 ∧ y = -2) ∨ (x = 3 ∧ y = 4))) → 
  m + b = -2 := by
  sorry

end line_slope_intercept_sum_l530_53008


namespace sufficient_but_not_necessary_l530_53092

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a = 1 → abs a = 1) ∧ ¬(abs a = 1 → a = 1) := by sorry

end sufficient_but_not_necessary_l530_53092


namespace second_die_sides_l530_53005

theorem second_die_sides (n : ℕ) : 
  n > 0 → 
  (1 : ℚ) / 6 * (1 : ℚ) / n = 0.023809523809523808 → 
  n = 7 := by
  sorry

end second_die_sides_l530_53005


namespace coach_mike_change_l530_53078

/-- The change received when paying for lemonade -/
def change (amount_paid : ℕ) (cost : ℕ) : ℕ :=
  amount_paid - cost

/-- Theorem: The change Coach Mike received is 17 cents -/
theorem coach_mike_change :
  change 75 58 = 17 := by
  sorry

end coach_mike_change_l530_53078


namespace johns_trip_duration_l530_53035

/-- Represents the duration of stay in weeks for each country visited -/
structure TripDuration where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total duration of a trip given the stay durations -/
def totalDuration (t : TripDuration) : ℕ :=
  t.first + t.second + t.third

/-- Theorem: The total duration of John's trip is 10 weeks -/
theorem johns_trip_duration :
  ∃ (t : TripDuration),
    t.first = 2 ∧
    t.second = 2 * t.first ∧
    t.third = 2 * t.first ∧
    totalDuration t = 10 :=
by
  sorry

end johns_trip_duration_l530_53035


namespace circle_symmetry_l530_53074

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 4)^2 + (y + 1)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ),
  original_circle x y ∧ symmetry_line x y →
  ∃ (x' y' : ℝ), symmetric_circle x' y' :=
by sorry

end circle_symmetry_l530_53074


namespace right_triangle_median_length_l530_53091

theorem right_triangle_median_length (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_sides : (a = 6 ∧ b = 8) ∨ (a = 6 ∧ c = 8) ∨ (b = 6 ∧ c = 8)) :
  ∃ m : ℝ, (m = 4 ∨ m = 5) ∧ 2 * m = c :=
sorry

end right_triangle_median_length_l530_53091


namespace min_distance_between_curves_l530_53056

open Real

theorem min_distance_between_curves : 
  ∀ (m n : ℝ), 
  2 * (m + 1) = n + log n → 
  |m - n| ≥ 3/2 ∧ 
  ∃ (m₀ n₀ : ℝ), 2 * (m₀ + 1) = n₀ + log n₀ ∧ |m₀ - n₀| = 3/2 := by
sorry

end min_distance_between_curves_l530_53056


namespace paving_stone_length_l530_53080

/-- Given a rectangular courtyard and paving stones with specific dimensions,
    calculate the length of each paving stone. -/
theorem paving_stone_length
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (total_stones : ℕ)
  (stone_width : ℝ)
  (h1 : courtyard_length = 30)
  (h2 : courtyard_width = 16)
  (h3 : total_stones = 240)
  (h4 : stone_width = 1)
  : (courtyard_length * courtyard_width) / (total_stones * stone_width) = 2 := by
  sorry

#check paving_stone_length

end paving_stone_length_l530_53080


namespace total_watermelons_is_48_l530_53090

/-- The number of watermelons Jason grew -/
def jason_watermelons : ℕ := 37

/-- The number of watermelons Sandy grew -/
def sandy_watermelons : ℕ := 11

/-- The total number of watermelons grown by Jason and Sandy -/
def total_watermelons : ℕ := jason_watermelons + sandy_watermelons

/-- Theorem stating that the total number of watermelons is 48 -/
theorem total_watermelons_is_48 : total_watermelons = 48 := by
  sorry

end total_watermelons_is_48_l530_53090


namespace abes_age_l530_53003

theorem abes_age (present_age : ℕ) : 
  present_age + (present_age - 7) = 29 → present_age = 18 := by
  sorry

end abes_age_l530_53003


namespace problem_solution_l530_53000

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 6 * x^2 + 12 * x * y + 6 * y^2 = x^3 + 3 * x^2 * y + 3 * x * y^2) :
  x = 24 / 7 := by
sorry

end problem_solution_l530_53000


namespace chick_hit_at_least_five_l530_53057

/-- Represents the number of times each toy was hit -/
structure ToyHits where
  chick : ℕ
  monkey : ℕ
  dog : ℕ

/-- Calculates the total score based on the number of hits for each toy -/
def calculateScore (hits : ToyHits) : ℕ :=
  9 * hits.chick + 5 * hits.monkey + 2 * hits.dog

/-- Checks if the given hits satisfy all conditions of the game -/
def isValidGame (hits : ToyHits) : Prop :=
  hits.chick > 0 ∧ hits.monkey > 0 ∧ hits.dog > 0 ∧
  hits.chick + hits.monkey + hits.dog = 10 ∧
  calculateScore hits = 61

/-- The minimum number of times the chick was hit in a valid game -/
def minChickHits : ℕ := 5

theorem chick_hit_at_least_five :
  ∀ hits : ToyHits, isValidGame hits → hits.chick ≥ minChickHits := by
  sorry

end chick_hit_at_least_five_l530_53057


namespace fifth_term_of_arithmetic_sequence_l530_53068

/-- Given an arithmetic sequence starting with 3, 7, 11, ..., 
    prove that its 5th term is 19. -/
theorem fifth_term_of_arithmetic_sequence : 
  ∀ (a : ℕ → ℝ), 
    (a 0 = 3) →  -- First term is 3
    (a 1 = 7) →  -- Second term is 7
    (a 2 = 11) → -- Third term is 11
    (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) → -- Arithmetic sequence property
    a 4 = 19 := by
  sorry

end fifth_term_of_arithmetic_sequence_l530_53068


namespace intersection_with_complement_l530_53071

open Set

def P : Set ℝ := {1, 2, 3, 4, 5}
def Q : Set ℝ := {4, 5, 6}

theorem intersection_with_complement : P ∩ (univ \ Q) = {1, 2, 3} := by
  sorry

end intersection_with_complement_l530_53071


namespace hyperbola_m_range_l530_53032

/-- Given a hyperbola with equation x^2 / (1+m) - y^2 / (3-m) = 1 and eccentricity > √2,
    prove that the range of m is (-1, 1) -/
theorem hyperbola_m_range (m : ℝ) :
  (∃ x y : ℝ, x^2 / (1 + m) - y^2 / (3 - m) = 1) ∧ 
  (2 / Real.sqrt (1 + m) > Real.sqrt 2) →
  -1 < m ∧ m < 1 :=
sorry

end hyperbola_m_range_l530_53032


namespace max_area_triangle_l530_53064

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition from the problem -/
def condition (t : Triangle) : Prop :=
  (Real.cos t.B / t.b) + (Real.cos t.C / t.c) = (2 * Real.sqrt 3 * Real.sin t.A) / (3 * Real.sin t.C)

/-- The theorem to be proved -/
theorem max_area_triangle (t : Triangle) (h1 : condition t) (h2 : t.B = Real.pi / 3) :
    (t.a * t.c * Real.sin t.B) / 2 ≤ 3 * Real.sqrt 3 / 16 := by
  sorry

end max_area_triangle_l530_53064


namespace arithmetic_square_root_of_16_l530_53041

theorem arithmetic_square_root_of_16 :
  Real.sqrt 16 = 4 := by
  sorry

end arithmetic_square_root_of_16_l530_53041


namespace square_difference_l530_53006

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 := by
  sorry

end square_difference_l530_53006


namespace plugged_handle_pressure_l530_53060

/-- The gauge pressure at the bottom of a jug with a plugged handle -/
theorem plugged_handle_pressure
  (ρ g h H P : ℝ)
  (h_pos : h > 0)
  (H_pos : H > 0)
  (H_gt_h : H > h)
  (ρ_pos : ρ > 0)
  (g_pos : g > 0) :
  ρ * g * H < P ∧ P < ρ * g * h :=
sorry

end plugged_handle_pressure_l530_53060


namespace f_2013_value_l530_53062

-- Define the properties of function f
def is_valid_f (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x, f (x + 3) = -f (1 - x)) ∧  -- given functional equation
  f 3 = 2  -- given value

-- State the theorem
theorem f_2013_value (f : ℝ → ℝ) (h : is_valid_f f) : f 2013 = -2 := by
  sorry

end f_2013_value_l530_53062


namespace replaced_person_weight_l530_53099

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℝ) (new_person_weight : ℝ) : ℝ :=
  new_person_weight - initial_count * average_increase

/-- Theorem stating that the weight of the replaced person is 60 kg -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 2.5 80 = 60 := by
  sorry

end replaced_person_weight_l530_53099


namespace dividend_calculation_l530_53085

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 19)
  (h2 : quotient = 7)
  (h3 : remainder = 6) :
  divisor * quotient + remainder = 139 := by
  sorry

end dividend_calculation_l530_53085


namespace standard_poodle_height_l530_53037

/-- The height of the toy poodle in inches -/
def toy_height : ℕ := 14

/-- The height difference between the miniature and toy poodle in inches -/
def mini_toy_diff : ℕ := 6

/-- The height difference between the standard and miniature poodle in inches -/
def standard_mini_diff : ℕ := 8

/-- The height of the standard poodle in inches -/
def standard_height : ℕ := toy_height + mini_toy_diff + standard_mini_diff

theorem standard_poodle_height : standard_height = 28 := by
  sorry

end standard_poodle_height_l530_53037


namespace regular_polygon_with_135_degree_angles_has_8_sides_l530_53097

/-- The number of sides of a regular polygon with interior angles measuring 135 degrees. -/
def regular_polygon_sides : ℕ := 8

/-- The measure of each interior angle of the regular polygon in degrees. -/
def interior_angle : ℝ := 135

/-- Theorem stating that a regular polygon with interior angles measuring 135 degrees has 8 sides. -/
theorem regular_polygon_with_135_degree_angles_has_8_sides :
  (interior_angle * regular_polygon_sides : ℝ) = 180 * (regular_polygon_sides - 2) :=
by sorry

end regular_polygon_with_135_degree_angles_has_8_sides_l530_53097


namespace triangle_property_l530_53036

theorem triangle_property (A B C : ℝ) (a b c : ℝ) :
  let triangle_exists := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b
  let vectors_dot_product := a * c * Real.cos B + 2 * b * c * Real.cos A = b * a * Real.cos C
  let side_angle_relation := 2 * a * Real.cos C = 2 * b - c
  triangle_exists →
  vectors_dot_product →
  side_angle_relation →
  Real.sin A / Real.sin C = Real.sqrt 2 ∧
  Real.cos B = (3 * Real.sqrt 2 - Real.sqrt 10) / 8 :=
by sorry

end triangle_property_l530_53036


namespace complex_equation_solution_l530_53066

theorem complex_equation_solution (a b : ℝ) :
  (Complex.mk a b) * (Complex.mk 2 (-1)) = Complex.I →
  a + b = 1/5 := by
sorry

end complex_equation_solution_l530_53066


namespace fraction_inequality_solution_set_l530_53040

theorem fraction_inequality_solution_set (x : ℝ) (h : x ≠ 0) :
  (x - 1) / x > 1 ↔ x < 0 := by
  sorry

end fraction_inequality_solution_set_l530_53040


namespace cricket_team_age_difference_l530_53009

def cricket_team_problem (team_size : ℕ) (avg_age : ℚ) (wicket_keeper_age_diff : ℚ) : Prop :=
  let total_age : ℚ := team_size * avg_age
  let wicket_keeper_age : ℚ := avg_age + wicket_keeper_age_diff
  let remaining_total_age : ℚ := total_age - wicket_keeper_age - avg_age
  let remaining_team_size : ℕ := team_size - 2
  let remaining_avg_age : ℚ := remaining_total_age / remaining_team_size
  (avg_age - remaining_avg_age) = 0.3

theorem cricket_team_age_difference :
  cricket_team_problem 11 24 3 := by
  sorry

end cricket_team_age_difference_l530_53009


namespace missing_chess_pieces_l530_53052

theorem missing_chess_pieces (total_pieces : Nat) (present_pieces : Nat) : 
  total_pieces = 32 → present_pieces = 22 → total_pieces - present_pieces = 10 := by
  sorry

end missing_chess_pieces_l530_53052


namespace higher_probability_white_piece_l530_53081

theorem higher_probability_white_piece (white_count black_count : ℕ) 
  (hw : white_count = 10) (hb : black_count = 2) : 
  (white_count : ℚ) / (white_count + black_count) > (black_count : ℚ) / (white_count + black_count) := by
  sorry

end higher_probability_white_piece_l530_53081


namespace at_least_one_girl_selection_l530_53039

theorem at_least_one_girl_selection (n_boys n_girls k : ℕ) 
  (h_boys : n_boys = 3) 
  (h_girls : n_girls = 4) 
  (h_select : k = 3) : 
  Nat.choose (n_boys + n_girls) k - Nat.choose n_boys k = 34 := by
  sorry

end at_least_one_girl_selection_l530_53039


namespace zoo_trip_cost_l530_53050

def zoo_ticket_cost : ℝ := 5
def bus_fare_one_way : ℝ := 1.5
def num_people : ℕ := 2
def lunch_snacks_money : ℝ := 24

def total_zoo_tickets : ℝ := zoo_ticket_cost * num_people
def total_bus_fare : ℝ := bus_fare_one_way * num_people * 2

theorem zoo_trip_cost : total_zoo_tickets + total_bus_fare + lunch_snacks_money = 40 := by
  sorry

end zoo_trip_cost_l530_53050


namespace PQ_perpendicular_RS_l530_53030

-- Define the points
variable (A B C D M P Q R S : ℝ × ℝ)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect_at (A B C D M : ℝ × ℝ) : Prop := sorry

-- Define centroid
def is_centroid (P A M D : ℝ × ℝ) : Prop := sorry

-- Define orthocenter
def is_orthocenter (R D M C : ℝ × ℝ) : Prop := sorry

-- Define perpendicularity of vectors
def vectors_perpendicular (P Q R S : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem PQ_perpendicular_RS 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : diagonals_intersect_at A B C D M)
  (h3 : is_centroid P A M D)
  (h4 : is_centroid Q C M B)
  (h5 : is_orthocenter R D M C)
  (h6 : is_orthocenter S M A B) :
  vectors_perpendicular P Q R S := by sorry

end PQ_perpendicular_RS_l530_53030


namespace female_democrat_ratio_l530_53025

theorem female_democrat_ratio (total_participants male_participants female_participants : ℕ)
  (female_democrats male_democrats : ℕ) :
  total_participants = 780 →
  total_participants = male_participants + female_participants →
  male_democrats = male_participants / 4 →
  female_democrats = 130 →
  male_democrats + female_democrats = total_participants / 3 →
  (female_democrats : ℚ) / female_participants = 1 / 2 :=
by sorry

end female_democrat_ratio_l530_53025


namespace y_decreases_as_x_increases_l530_53007

def linear_function (x : ℝ) : ℝ := -3 * x + 6

theorem y_decreases_as_x_increases :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → linear_function x₁ > linear_function x₂ := by
  sorry

end y_decreases_as_x_increases_l530_53007


namespace marble_box_count_l530_53026

theorem marble_box_count : ∀ (p y u : ℕ),
  y + u = 10 →
  p + u = 12 →
  p + y = 6 →
  p + y + u = 14 :=
by
  sorry

end marble_box_count_l530_53026


namespace joes_steakhouse_wage_difference_l530_53031

/-- Proves the wage difference between a manager and a chef at Joe's Steakhouse -/
theorem joes_steakhouse_wage_difference :
  let manager_wage : ℚ := 85/10
  let dishwasher_wage : ℚ := manager_wage / 2
  let chef_wage : ℚ := dishwasher_wage * (1 + 1/4)
  manager_wage - chef_wage = 3187/1000 := by
  sorry

end joes_steakhouse_wage_difference_l530_53031


namespace part1_part2_l530_53028

-- Define the functions f and h
def f (m : ℝ) (x : ℝ) : ℝ := |x - m|
def h (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem for part 1
theorem part1 (m : ℝ) : 
  (∀ x, f m x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → m = 2 :=
sorry

-- Theorem for part 2
theorem part2 (t : ℝ) :
  (∃ x, x^2 + 6*x + h t = 0) → -5 ≤ t ∧ t ≤ 4 :=
sorry

end part1_part2_l530_53028


namespace older_females_count_l530_53002

/-- Represents the population of a town divided into equal groups -/
structure TownPopulation where
  total : ℕ
  num_groups : ℕ
  h_positive : 0 < num_groups

/-- Calculates the size of each group in the town -/
def group_size (town : TownPopulation) : ℕ :=
  town.total / town.num_groups

/-- Theorem: In a town with 1000 people divided into 5 equal groups,
    the number of people in each group is 200 -/
theorem older_females_count (town : TownPopulation)
    (h_total : town.total = 1000)
    (h_groups : town.num_groups = 5) :
    group_size town = 200 := by
  sorry

#eval group_size ⟨1000, 5, by norm_num⟩

end older_females_count_l530_53002


namespace polynomial_division_proof_l530_53077

theorem polynomial_division_proof :
  ∀ (x : ℚ),
  (3*x + 1) * (2*x^3 + x^2 - 7/3*x + 20/9) + 31/27 = 6*x^4 + 5*x^3 - 4*x^2 + x + 1 := by
  sorry

end polynomial_division_proof_l530_53077


namespace root_difference_zero_l530_53027

theorem root_difference_zero (x : ℝ) : 
  (x^2 + 40*x + 300 = -100) → 
  (∃ r₁ r₂ : ℝ, (r₁^2 + 40*r₁ + 300 = -100) ∧ 
                (r₂^2 + 40*r₂ + 300 = -100) ∧ 
                (|r₁ - r₂| = 0)) := by
  sorry

end root_difference_zero_l530_53027


namespace survey_participants_survey_participants_proof_l530_53053

theorem survey_participants (total_participants : ℕ) 
  (first_myth_percentage : ℚ) 
  (second_myth_percentage : ℚ) 
  (both_myths_count : ℕ) : Prop :=
  first_myth_percentage = 923 / 1000 ∧ 
  second_myth_percentage = 382 / 1000 ∧
  both_myths_count = 29 →
  total_participants = 83

-- The proof of the theorem
theorem survey_participants_proof : 
  ∃ (total_participants : ℕ), 
    survey_participants total_participants (923 / 1000) (382 / 1000) 29 :=
by
  sorry

end survey_participants_survey_participants_proof_l530_53053


namespace special_hexagon_area_l530_53010

/-- A hexagon with specific properties -/
structure SpecialHexagon where
  -- Each angle measures 120°
  angle_measure : ℝ
  angle_measure_eq : angle_measure = 120
  -- Sides alternately measure 1 cm and √3 cm
  side_length1 : ℝ
  side_length2 : ℝ
  side_length1_eq : side_length1 = 1
  side_length2_eq : side_length2 = Real.sqrt 3

/-- The area of the special hexagon -/
noncomputable def area (h : SpecialHexagon) : ℝ := 3 + Real.sqrt 3

/-- Theorem stating that the area of the special hexagon is 3 + √3 cm² -/
theorem special_hexagon_area (h : SpecialHexagon) : area h = 3 + Real.sqrt 3 := by
  sorry

end special_hexagon_area_l530_53010


namespace extra_distance_for_early_arrival_l530_53065

theorem extra_distance_for_early_arrival
  (S : ℝ) -- distance between A and B in kilometers
  (a : ℝ) -- original planned arrival time in hours
  (h : a > 2) -- condition that a > 2
  : (S / (a - 2) - S / a) = -- extra distance per hour needed for early arrival
    (S / (a - 2) - S / a) :=
by sorry

end extra_distance_for_early_arrival_l530_53065


namespace complex_equation_solution_l530_53048

theorem complex_equation_solution :
  ∀ z : ℂ, (1 + 2*I) * z = 1 - I → z = -1/5 - 3/5*I :=
by
  sorry

end complex_equation_solution_l530_53048


namespace spade_problem_l530_53016

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_problem : spade 5 (spade 3 2) = 0 := by
  sorry

end spade_problem_l530_53016


namespace statement_a_correct_statement_b_correct_statement_c_incorrect_statement_d_correct_main_theorem_l530_53076

-- Statement A
theorem statement_a_correct (x y a : ℝ) : x^2 = y^2 → -3*a*x^2 = -3*a*y^2 := by sorry

-- Statement B
theorem statement_b_correct (x y a : ℝ) (h : a ≠ 0) : x / a = y / a → x = y := by sorry

-- Statement C (incorrect)
theorem statement_c_incorrect : ∃ a b c : ℝ, a*c = b*c ∧ a ≠ b := by sorry

-- Statement D
theorem statement_d_correct (a b : ℝ) : a = b → a^2 = b^2 := by sorry

-- Main theorem
theorem main_theorem : 
  (∀ x y a : ℝ, x^2 = y^2 → -3*a*x^2 = -3*a*y^2) ∧
  (∀ x y a : ℝ, a ≠ 0 → (x / a = y / a → x = y)) ∧
  (∃ a b c : ℝ, a*c = b*c ∧ a ≠ b) ∧
  (∀ a b : ℝ, a = b → a^2 = b^2) := by sorry

end statement_a_correct_statement_b_correct_statement_c_incorrect_statement_d_correct_main_theorem_l530_53076


namespace arithmetic_sequence_property_l530_53013

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The sum of specific terms in the sequence equals 120 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) :
  2 * a 10 - a 12 = 24 := by
  sorry

end arithmetic_sequence_property_l530_53013


namespace largest_fraction_l530_53018

theorem largest_fraction
  (a b c d e : ℝ)
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : b < c)
  (h4 : c < d)
  (h5 : d < e) :
  (d + e) / (a + b) > (a + c) / (b + d) ∧
  (d + e) / (a + b) > (b + e) / (c + d) ∧
  (d + e) / (a + b) > (c + d) / (a + e) ∧
  (d + e) / (a + b) > (e + a) / (b + c) :=
by sorry


end largest_fraction_l530_53018


namespace circle_intersection_r_range_l530_53034

/-- Two circles have a common point if and only if the distance between their centers
    is between the difference and sum of their radii. -/
axiom circles_intersect (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ) :
  ∃ (p : ℝ × ℝ), (p.1 - c₁.1)^2 + (p.2 - c₁.2)^2 = r₁^2 ∧
                 (p.1 - c₂.1)^2 + (p.2 - c₂.2)^2 = r₂^2 ↔
  |r₁ - r₂| ≤ Real.sqrt ((c₁.1 - c₂.1)^2 + (c₁.2 - c₂.2)^2) ∧
  Real.sqrt ((c₁.1 - c₂.1)^2 + (c₁.2 - c₂.2)^2) ≤ r₁ + r₂

/-- The theorem stating the range of r for two intersecting circles -/
theorem circle_intersection_r_range :
  ∀ r : ℝ, r > 0 →
  (∃ (p : ℝ × ℝ), p.1^2 + p.2^2 = r^2 ∧ (p.1 - 3)^2 + (p.2 + 4)^2 = 49) →
  2 ≤ r ∧ r ≤ 12 := by
  sorry

end circle_intersection_r_range_l530_53034


namespace arithmetic_sequence_geometric_mean_l530_53049

/-- An arithmetic sequence with common difference d and first term 9d -/
def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := 9 * d + (n - 1) * d

theorem arithmetic_sequence_geometric_mean (d : ℝ) (k : ℕ) 
  (h_d : d ≠ 0) :
  (arithmetic_sequence d k) ^ 2 = 
    (arithmetic_sequence d 1) * (arithmetic_sequence d (2 * k)) → 
  k = 4 := by
  sorry

end arithmetic_sequence_geometric_mean_l530_53049


namespace orange_min_cost_l530_53083

/-- Represents the cost and quantity of oranges in a package -/
structure Package where
  quantity : ℕ
  cost : ℕ

/-- Calculates the minimum cost to buy a given number of oranges -/
def minCost (bag : Package) (box : Package) (total : ℕ) : ℕ :=
  sorry

theorem orange_min_cost :
  let bag : Package := ⟨4, 12⟩
  let box : Package := ⟨6, 25⟩
  let total : ℕ := 20
  minCost bag box total = 60 := by
  sorry

end orange_min_cost_l530_53083


namespace total_guppies_l530_53059

/-- The number of guppies owned by each person -/
structure GuppyOwners where
  haylee : ℕ
  jose : ℕ
  charliz : ℕ
  nicolai : ℕ
  alice : ℕ
  bob : ℕ
  cameron : ℕ

/-- The conditions of guppy ownership as described in the problem -/
def guppy_conditions (g : GuppyOwners) : Prop :=
  g.haylee = 3 * 12 ∧
  g.jose = g.haylee / 2 ∧
  g.charliz = g.jose / 3 ∧
  g.nicolai = 4 * g.charliz ∧
  g.alice = g.nicolai + 5 ∧
  g.bob = (g.jose + g.charliz) / 2 ∧
  g.cameron = 2^3

/-- The theorem stating that the total number of guppies is 133 -/
theorem total_guppies (g : GuppyOwners) (h : guppy_conditions g) :
  g.haylee + g.jose + g.charliz + g.nicolai + g.alice + g.bob + g.cameron = 133 := by
  sorry


end total_guppies_l530_53059


namespace oil_price_relation_l530_53055

/-- Represents a right circular cylinder -/
structure Cylinder where
  height : ℝ
  radius : ℝ

/-- The price of oil in a cylinder when filled to a certain capacity -/
def oilPrice (c : Cylinder) (fillRatio : ℝ) : ℝ := sorry

theorem oil_price_relation (x y : Cylinder) :
  y.height = 4 * x.height →
  y.radius = 4 * x.radius →
  oilPrice y 0.5 = 64 →
  oilPrice x 1 = 2 := by
  sorry

end oil_price_relation_l530_53055


namespace right_triangle_shorter_leg_l530_53096

theorem right_triangle_shorter_leg : ∀ a b c : ℕ,
  a * a + b * b = c * c →  -- Pythagorean theorem
  c = 65 →                 -- hypotenuse length
  a ≤ b →                  -- a is the shorter leg
  a = 25 := by
sorry

end right_triangle_shorter_leg_l530_53096


namespace degree_of_specific_monomial_l530_53047

def monomial_degree (coeff : ℤ) (vars : List (Char × ℕ)) : ℕ :=
  (vars.map (·.2)).sum

theorem degree_of_specific_monomial :
  monomial_degree (-5) [('a', 2), ('b', 3)] = 5 := by
  sorry

end degree_of_specific_monomial_l530_53047


namespace square_sum_zero_implies_both_zero_l530_53011

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end square_sum_zero_implies_both_zero_l530_53011


namespace power_of_three_mod_ten_l530_53072

theorem power_of_three_mod_ten : 3^19 % 10 = 7 := by sorry

end power_of_three_mod_ten_l530_53072


namespace first_day_exceeding_500_l530_53004

def bacteria_growth (n : ℕ) : ℕ := 5 * 3^n

theorem first_day_exceeding_500 :
  (∀ k < 6, bacteria_growth k ≤ 500) ∧ bacteria_growth 6 > 500 := by sorry

end first_day_exceeding_500_l530_53004


namespace arithmetic_sequence_sum_l530_53095

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) (h1 : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 := by
  sorry

end arithmetic_sequence_sum_l530_53095


namespace abes_family_total_yen_l530_53042

/-- Given Abe's family's checking and savings account balances in yen, 
    prove that their total amount of yen is 9844. -/
theorem abes_family_total_yen (checking : ℕ) (savings : ℕ) 
    (h1 : checking = 6359) (h2 : savings = 3485) : 
    checking + savings = 9844 := by
  sorry

end abes_family_total_yen_l530_53042


namespace inequality_satisfied_l530_53020

theorem inequality_satisfied (a b c : ℤ) : 
  a = 1 ∧ b = 2 ∧ c = 1 → a^2 + b^2 + c^2 + 3 < a*b + 3*b + 2*c := by sorry

end inequality_satisfied_l530_53020


namespace notebook_marker_cost_l530_53012

theorem notebook_marker_cost (notebook_cost marker_cost : ℝ) 
  (h1 : 3 * notebook_cost + 2 * marker_cost = 7.20)
  (h2 : 2 * notebook_cost + 3 * marker_cost = 6.90) :
  notebook_cost + marker_cost = 2.82 := by
  sorry

end notebook_marker_cost_l530_53012


namespace vegetable_planting_methods_l530_53033

theorem vegetable_planting_methods (n : Nat) (k : Nat) (m : Nat) : 
  n = 4 ∧ k = 3 ∧ m = 3 → 
  (Nat.choose (n - 1) (k - 1)) * (Nat.factorial k) = 18 :=
by
  sorry

end vegetable_planting_methods_l530_53033
