import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l218_21879

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 + a 4 + a 10 + a 16 + a 19 = 150 →
  a 20 - a 26 + a 16 = 30 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l218_21879


namespace NUMINAMATH_CALUDE_parabola_min_value_l218_21883

theorem parabola_min_value (x y : ℝ) : 
  y^2 = 4*x → (∀ x' y' : ℝ, y'^2 = 4*x' → 1/2 * y'^2 + x'^2 + 3 ≥ 1/2 * y^2 + x^2 + 3) → 
  1/2 * y^2 + x^2 + 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_min_value_l218_21883


namespace NUMINAMATH_CALUDE_square_difference_l218_21815

theorem square_difference : (169 * 169) - (168 * 168) = 337 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l218_21815


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l218_21845

/-- The function f(x) = a^(x-2) + 2 has a fixed point at (2, 3) for a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 2
  f 2 = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l218_21845


namespace NUMINAMATH_CALUDE_negative_values_iff_a_outside_interval_l218_21819

/-- A quadratic function f(x) = x^2 - ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

/-- The function f takes negative values -/
def takes_negative_values (a : ℝ) : Prop :=
  ∃ x, f a x < 0

/-- The main theorem: f takes negative values iff a > 2 or a < -2 -/
theorem negative_values_iff_a_outside_interval :
  ∀ a : ℝ, takes_negative_values a ↔ (a > 2 ∨ a < -2) :=
sorry

end NUMINAMATH_CALUDE_negative_values_iff_a_outside_interval_l218_21819


namespace NUMINAMATH_CALUDE_mechanic_work_hours_l218_21861

/-- Calculates the number of hours a mechanic worked given the total cost, part costs, and labor rate. -/
theorem mechanic_work_hours 
  (total_cost : ℝ) 
  (part_cost : ℝ) 
  (labor_rate_per_minute : ℝ) 
  (h1 : total_cost = 220) 
  (h2 : part_cost = 20) 
  (h3 : labor_rate_per_minute = 0.5) : 
  (total_cost - 2 * part_cost) / (labor_rate_per_minute * 60) = 6 := by
  sorry

end NUMINAMATH_CALUDE_mechanic_work_hours_l218_21861


namespace NUMINAMATH_CALUDE_games_per_season_l218_21822

/-- Given the following conditions:
  - Louie scored 4 goals in the last match
  - Louie scored 40 goals in previous matches
  - Louie's brother scored twice as many goals as Louie in the last match
  - Louie's brother has played for 3 seasons
  - The total number of goals scored by both brothers is 1244
Prove that there are 50 games in each season -/
theorem games_per_season (louie_last_match : ℕ) (louie_previous : ℕ) 
  (brother_multiplier : ℕ) (brother_seasons : ℕ) (total_goals : ℕ) :
  louie_last_match = 4 →
  louie_previous = 40 →
  brother_multiplier = 2 →
  brother_seasons = 3 →
  total_goals = 1244 →
  ∃ (games_per_season : ℕ), 
    louie_last_match + louie_previous + 
    brother_multiplier * louie_last_match * games_per_season * brother_seasons = 
    total_goals ∧ games_per_season = 50 :=
by sorry

end NUMINAMATH_CALUDE_games_per_season_l218_21822


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l218_21890

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l218_21890


namespace NUMINAMATH_CALUDE_circle_power_theorem_l218_21813

/-- Given a circle with center O and radius R, and points A and B on the circle,
    for any point P on line AB, PA * PB = OP^2 - R^2 in terms of algebraic lengths -/
theorem circle_power_theorem (O : ℝ × ℝ) (R : ℝ) (A B P : ℝ × ℝ) :
  (∀ X : ℝ × ℝ, dist O X = R → (X = A ∨ X = B)) →
  (∃ t : ℝ, P = (1 - t) • A + t • B) →
  (dist P A * dist P B : ℝ) = dist O P ^ 2 - R ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_power_theorem_l218_21813


namespace NUMINAMATH_CALUDE_inverse_square_direct_cube_relation_l218_21807

/-- Given that x varies inversely as the square of y and directly as the cube of z,
    prove that when x = 1 for y = 3 and z = 2, then x = 8/9 when y = 9 and z = 4. -/
theorem inverse_square_direct_cube_relation (k : ℚ) :
  (1 : ℚ) = k * (2^3 : ℚ) / (3^2 : ℚ) →
  (8/9 : ℚ) = k * (4^3 : ℚ) / (9^2 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_inverse_square_direct_cube_relation_l218_21807


namespace NUMINAMATH_CALUDE_line_through_points_with_xintercept_l218_21859

/-- A line in a 2D plane -/
structure Line where
  slope : ℚ
  yIntercept : ℚ

/-- Create a line from two points -/
def Line.fromPoints (x1 y1 x2 y2 : ℚ) : Line :=
  let slope := (y2 - y1) / (x2 - x1)
  let yIntercept := y1 - slope * x1
  { slope := slope, yIntercept := yIntercept }

/-- Get the x-coordinate for a given y-coordinate on a line -/
def Line.xCoordinate (l : Line) (y : ℚ) : ℚ :=
  (y - l.yIntercept) / l.slope

theorem line_through_points_with_xintercept
  (line : Line)
  (h1 : line = Line.fromPoints 4 0 10 3)
  (h2 : line.xCoordinate (-6) = -8) : True :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_with_xintercept_l218_21859


namespace NUMINAMATH_CALUDE_license_plate_count_l218_21844

/-- The number of possible first letters for the license plate -/
def first_letter_choices : ℕ := 3

/-- The number of choices for each digit position -/
def digit_choices : ℕ := 10

/-- The number of digit positions after the letter -/
def num_digits : ℕ := 5

/-- The total number of possible license plates -/
def total_license_plates : ℕ := first_letter_choices * digit_choices ^ num_digits

theorem license_plate_count : total_license_plates = 300000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l218_21844


namespace NUMINAMATH_CALUDE_yasmin_has_two_children_l218_21838

/-- The number of children Yasmin has -/
def yasmin_children : ℕ := 2

/-- The number of children John has -/
def john_children : ℕ := 2 * yasmin_children

/-- The total number of grandchildren -/
def total_grandchildren : ℕ := 6

theorem yasmin_has_two_children :
  yasmin_children = 2 ∧
  john_children = 2 * yasmin_children ∧
  yasmin_children + john_children = total_grandchildren :=
sorry

end NUMINAMATH_CALUDE_yasmin_has_two_children_l218_21838


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l218_21886

/-- The number of dots in each row and column of the square array -/
def gridSize : Nat := 5

/-- The number of different rectangles that can be formed in the grid -/
def numRectangles : Nat := (gridSize.choose 2) * (gridSize.choose 2)

/-- Theorem stating the number of rectangles in a 5x5 grid -/
theorem rectangles_in_5x5_grid : numRectangles = 100 := by sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l218_21886


namespace NUMINAMATH_CALUDE_line_vertical_translation_l218_21818

/-- The equation of a line after vertical translation -/
theorem line_vertical_translation (x y : ℝ) :
  (y = x) → (y = x + 2) ↔ (∀ point : ℝ × ℝ, point.2 = point.1 + 2 ↔ point.2 = point.1 + 2) :=
by sorry

end NUMINAMATH_CALUDE_line_vertical_translation_l218_21818


namespace NUMINAMATH_CALUDE_point_P_on_circle_O_l218_21875

/-- A circle with center at the origin and radius 5 -/
def circle_O : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 25}

/-- Point P with coordinates (4,3) -/
def point_P : ℝ × ℝ := (4, 3)

/-- Theorem stating that point P lies on circle O -/
theorem point_P_on_circle_O : point_P ∈ circle_O := by
  sorry

end NUMINAMATH_CALUDE_point_P_on_circle_O_l218_21875


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l218_21843

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (0 < x ∧ x < 4) → (x^2 - 3*x < 0 → 0 < x ∧ x < 4) ∧ ¬(0 < x ∧ x < 4 → x^2 - 3*x < 0) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l218_21843


namespace NUMINAMATH_CALUDE_total_score_is_38_l218_21804

/-- Represents the scores of three friends in a table football game. -/
structure Scores where
  darius : ℕ
  matt : ℕ
  marius : ℕ

/-- The conditions of the game and the total score calculation. -/
def game_result (s : Scores) : Prop :=
  s.marius = s.darius + 3 ∧
  s.matt = s.darius + 5 ∧
  s.darius = 10 ∧
  s.darius + s.matt + s.marius = 38

/-- Theorem stating that under the given conditions, the total score is 38. -/
theorem total_score_is_38 : ∃ s : Scores, game_result s :=
  sorry

end NUMINAMATH_CALUDE_total_score_is_38_l218_21804


namespace NUMINAMATH_CALUDE_cube_side_area_l218_21882

theorem cube_side_area (edge_sum : ℝ) (h : edge_sum = 132) : 
  let edge_length := edge_sum / 12
  (edge_length ^ 2) = 121 := by sorry

end NUMINAMATH_CALUDE_cube_side_area_l218_21882


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l218_21860

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l218_21860


namespace NUMINAMATH_CALUDE_rope_pieces_needed_l218_21877

/-- The number of stories Tom needs to lower the rope --/
def stories : ℕ := 6

/-- The height of one story in feet --/
def story_height : ℕ := 10

/-- The length of one piece of rope in feet --/
def rope_length : ℕ := 20

/-- The percentage of rope lost when lashing pieces together --/
def rope_loss_percentage : ℚ := 1/4

/-- The number of pieces of rope Tom needs to buy --/
def pieces_needed : ℕ := 4

theorem rope_pieces_needed :
  (stories * story_height : ℚ) ≤ pieces_needed * (rope_length * (1 - rope_loss_percentage)) ∧
  (stories * story_height : ℚ) > (pieces_needed - 1) * (rope_length * (1 - rope_loss_percentage)) :=
sorry

end NUMINAMATH_CALUDE_rope_pieces_needed_l218_21877


namespace NUMINAMATH_CALUDE_tims_soda_cans_l218_21867

theorem tims_soda_cans (x : ℕ) : 
  x - 6 + (x - 6) / 2 = 24 → x = 22 := by
  sorry

end NUMINAMATH_CALUDE_tims_soda_cans_l218_21867


namespace NUMINAMATH_CALUDE_green_paint_calculation_l218_21837

/-- Given a paint mixture ratio and the amount of white paint, 
    calculate the amount of green paint needed. -/
theorem green_paint_calculation 
  (blue green white : ℚ) 
  (ratio : blue / green = 5 / 3 ∧ green / white = 3 / 7) 
  (white_amount : white = 21) : 
  green = 9 := by
sorry

end NUMINAMATH_CALUDE_green_paint_calculation_l218_21837


namespace NUMINAMATH_CALUDE_expand_expression_l218_21826

theorem expand_expression (x : ℝ) : (x - 3) * (x + 6) = x^2 + 3*x - 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l218_21826


namespace NUMINAMATH_CALUDE_star_two_neg_three_l218_21841

/-- The ⋆ operation defined on real numbers -/
def star (a b : ℝ) : ℝ := a^2 * b^2 + a - 1

/-- Theorem stating that 2 ⋆ (-3) = 37 -/
theorem star_two_neg_three : star 2 (-3) = 37 := by
  sorry

end NUMINAMATH_CALUDE_star_two_neg_three_l218_21841


namespace NUMINAMATH_CALUDE_kenneth_distance_past_finish_line_l218_21862

/-- A proof that Kenneth will be 10 yards past the finish line when Biff crosses it in a 500-yard race -/
theorem kenneth_distance_past_finish_line 
  (race_distance : ℝ) 
  (biff_speed : ℝ) 
  (kenneth_speed : ℝ) 
  (h1 : race_distance = 500)
  (h2 : biff_speed = 50)
  (h3 : kenneth_speed = 51) :
  kenneth_speed * (race_distance / biff_speed) - race_distance = 10 :=
by
  sorry

#check kenneth_distance_past_finish_line

end NUMINAMATH_CALUDE_kenneth_distance_past_finish_line_l218_21862


namespace NUMINAMATH_CALUDE_problem_solution_l218_21810

def δ (x : ℝ) : ℝ := 3 * x + 8
def φ (x : ℝ) : ℝ := 9 * x + 7

theorem problem_solution (x : ℝ) : δ (φ x) = 11 → x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l218_21810


namespace NUMINAMATH_CALUDE_certain_number_multiplication_l218_21864

theorem certain_number_multiplication (x : ℚ) : x / 11 = 2 → 6 * x = 132 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_multiplication_l218_21864


namespace NUMINAMATH_CALUDE_trick_decks_spending_l218_21812

/-- The total amount spent by Frank and his friend on trick decks -/
def total_spent (deck_price : ℕ) (frank_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  deck_price * frank_decks + deck_price * friend_decks

/-- Theorem: Frank and his friend spent 35 dollars on trick decks -/
theorem trick_decks_spending : total_spent 7 3 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_trick_decks_spending_l218_21812


namespace NUMINAMATH_CALUDE_white_dandelions_on_saturday_l218_21828

/-- Represents the day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the state of a dandelion -/
inductive DandelionState
  | Yellow
  | White
  | Dispersed

/-- Represents the count of dandelions in different states -/
structure DandelionCount where
  yellow : ℕ
  white : ℕ

/-- The life cycle of a dandelion -/
def dandelionLifeCycle (day : ℕ) : DandelionState :=
  match day with
  | 0 | 1 | 2 => DandelionState.Yellow
  | 3 | 4 => DandelionState.White
  | _ => DandelionState.Dispersed

/-- Count of dandelions on a given day -/
def dandelionCountOnDay (day : Day) : DandelionCount :=
  match day with
  | Day.Monday => { yellow := 20, white := 14 }
  | Day.Wednesday => { yellow := 15, white := 11 }
  | _ => { yellow := 0, white := 0 }  -- We don't have information for other days

/-- Days between two given days -/
def daysBetween (start finish : Day) : ℕ :=
  match start, finish with
  | Day.Monday, Day.Wednesday => 2
  | Day.Wednesday, Day.Saturday => 3
  | _, _ => 0  -- We don't need other cases for this problem

/-- The main theorem -/
theorem white_dandelions_on_saturday :
  ∃ (new_dandelions : ℕ),
    new_dandelions = (dandelionCountOnDay Day.Wednesday).yellow + (dandelionCountOnDay Day.Wednesday).white
                   - (dandelionCountOnDay Day.Monday).yellow
    ∧ new_dandelions = 6
    ∧ (dandelionLifeCycle (daysBetween Day.Tuesday Day.Saturday) = DandelionState.White
    ∧ dandelionLifeCycle (daysBetween Day.Wednesday Day.Saturday) = DandelionState.White)
    → new_dandelions = 6 := by sorry


end NUMINAMATH_CALUDE_white_dandelions_on_saturday_l218_21828


namespace NUMINAMATH_CALUDE_planar_graph_iff_euler_l218_21878

/-- A planar graph is a graph that can be embedded in the plane without edge crossings. -/
structure PlanarGraph where
  v : ℕ  -- number of vertices
  g : ℕ  -- number of edges
  s : ℕ  -- number of faces

/-- Euler's formula for planar graphs states that v - g + s = 2 -/
def satisfiesEulersFormula (graph : PlanarGraph) : Prop :=
  graph.v - graph.g + graph.s = 2

/-- A planar graph can be constructed if and only if it satisfies Euler's formula -/
theorem planar_graph_iff_euler (graph : PlanarGraph) :
  ∃ (G : PlanarGraph), G.v = graph.v ∧ G.g = graph.g ∧ G.s = graph.s ↔ satisfiesEulersFormula graph :=
sorry

end NUMINAMATH_CALUDE_planar_graph_iff_euler_l218_21878


namespace NUMINAMATH_CALUDE_greater_solution_quadratic_l218_21814

theorem greater_solution_quadratic : 
  ∃ (x : ℝ), x^2 + 14*x - 88 = 0 ∧ 
  (∀ (y : ℝ), y^2 + 14*y - 88 = 0 → y ≤ x) ∧
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_greater_solution_quadratic_l218_21814


namespace NUMINAMATH_CALUDE_sled_distance_l218_21800

/-- Represents the distance traveled by a sled in a given second -/
def distance_in_second (n : ℕ) : ℕ := 6 + (n - 1) * 8

/-- Calculates the total distance traveled by the sled over a given number of seconds -/
def total_distance (seconds : ℕ) : ℕ :=
  (seconds * (distance_in_second 1 + distance_in_second seconds)) / 2

/-- Theorem stating that a sled sliding for 20 seconds travels 1640 inches -/
theorem sled_distance : total_distance 20 = 1640 := by
  sorry

end NUMINAMATH_CALUDE_sled_distance_l218_21800


namespace NUMINAMATH_CALUDE_add_2057_minutes_to_3_15pm_l218_21854

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

theorem add_2057_minutes_to_3_15pm (start : Time) (result : Time) :
  start.hours = 15 ∧ start.minutes = 15 →
  result = addMinutes start 2057 →
  result.hours = 1 ∧ result.minutes = 32 := by
  sorry

end NUMINAMATH_CALUDE_add_2057_minutes_to_3_15pm_l218_21854


namespace NUMINAMATH_CALUDE_equation_solution_l218_21888

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (9 / x^2) - (6 / x) + 1 = 0 → 2 / x = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l218_21888


namespace NUMINAMATH_CALUDE_work_completion_days_l218_21892

/-- The number of days it takes person A to complete the work -/
def days_A : ℝ := 20

/-- The number of days it takes person B to complete the work -/
def days_B : ℝ := 30

/-- The number of days A worked before leaving -/
def days_A_worked : ℝ := 10

/-- The number of days B worked to finish the remaining work -/
def days_B_worked : ℝ := 15

/-- Theorem stating that A can complete the work in 20 days -/
theorem work_completion_days :
  (days_A_worked / days_A) + (days_B_worked / days_B) = 1 :=
sorry

end NUMINAMATH_CALUDE_work_completion_days_l218_21892


namespace NUMINAMATH_CALUDE_horner_v3_value_l218_21829

def horner_polynomial (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_step (v : ℝ) (x : ℝ) (a : ℝ) : ℝ := v * x + a

theorem horner_v3_value :
  let x := 2
  let v0 := 1
  let v1 := horner_step v0 x (-12)
  let v2 := horner_step v1 x 60
  let v3 := horner_step v2 x (-160)
  v3 = -80 := by sorry

end NUMINAMATH_CALUDE_horner_v3_value_l218_21829


namespace NUMINAMATH_CALUDE_subtract_negative_one_l218_21821

theorem subtract_negative_one : 3 - (-1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_one_l218_21821


namespace NUMINAMATH_CALUDE_gina_money_problem_l218_21808

theorem gina_money_problem (initial_amount : ℚ) : 
  initial_amount = 400 → 
  initial_amount - (initial_amount * (1/4 + 1/8 + 1/5)) = 170 := by
  sorry

end NUMINAMATH_CALUDE_gina_money_problem_l218_21808


namespace NUMINAMATH_CALUDE_brothers_age_ratio_l218_21806

/-- Represents the ages of three brothers: Richard, David, and Scott -/
structure BrothersAges where
  david : ℕ
  richard : ℕ
  scott : ℕ

/-- Calculates the ages of the brothers after a given number of years -/
def agesAfterYears (ages : BrothersAges) (years : ℕ) : BrothersAges :=
  { david := ages.david + years
  , richard := ages.richard + years
  , scott := ages.scott + years }

/-- The theorem statement based on the given problem -/
theorem brothers_age_ratio : ∀ (ages : BrothersAges),
  ages.richard = ages.david + 6 →
  ages.david = ages.scott + 8 →
  ages.david = 14 →
  ∃ (k : ℕ), (agesAfterYears ages 8).richard = k * (agesAfterYears ages 8).scott →
  (agesAfterYears ages 8).richard / (agesAfterYears ages 8).scott = 2 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_ratio_l218_21806


namespace NUMINAMATH_CALUDE_polygon_sides_l218_21889

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 1260 → ∃ n : ℕ, n = 9 ∧ sum_interior_angles = 180 * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l218_21889


namespace NUMINAMATH_CALUDE_line_circle_intersection_l218_21870

/-- Given a circle with radius 5 and a line at distance k from its center,
    if the equation x^2 - kx + 1 = 0 has equal roots,
    then the line intersects the circle. -/
theorem line_circle_intersection (k : ℝ) : 
  (∃ x : ℝ, x^2 - k*x + 1 = 0 ∧ (∀ y : ℝ, y^2 - k*y + 1 = 0 → y = x)) →
  k < 5 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l218_21870


namespace NUMINAMATH_CALUDE_f_inequality_l218_21895

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f x = f (x + p)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

def is_symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem f_inequality (f : ℝ → ℝ) 
  (h1 : is_periodic f 4)
  (h2 : is_increasing_on f 0 2)
  (h3 : is_symmetric_about (fun x ↦ f (x + 2)) 0) :
  f 4.5 < f 7 ∧ f 7 < f 6.5 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l218_21895


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l218_21858

/-- The original expression as a function of x and square -/
def original_expr (x : ℝ) (square : ℝ) : ℝ :=
  (3 - 2*x^2 - 5*x) - (square*x^2 + 3*x - 4)

/-- The simplified expression as a function of x and square -/
def simplified_expr (x : ℝ) (square : ℝ) : ℝ :=
  (-2 - square)*x^2 - 8*x + 7

theorem expression_simplification_and_evaluation :
  ∀ (x : ℝ) (square : ℝ),
  /- 1. The simplified form is correct -/
  original_expr x square = simplified_expr x square ∧
  /- 2. When x=-2 and square=-2, the expression evaluates to -17 -/
  original_expr (-2) (-2) = -17 ∧
  /- 3. The value of square that eliminates the quadratic term is -2 -/
  ∃ (square : ℝ), (-2 - square) = 0 ∧ square = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l218_21858


namespace NUMINAMATH_CALUDE_currency_conversion_l218_21817

-- Define the conversion rates
def cents_per_jiao : ℝ := 10
def cents_per_yuan : ℝ := 100

-- Define the theorem
theorem currency_conversion :
  (5 / cents_per_jiao = 0.5) ∧ 
  (5 / cents_per_yuan = 0.05) ∧ 
  (3.25 * cents_per_yuan = 325) := by
sorry


end NUMINAMATH_CALUDE_currency_conversion_l218_21817


namespace NUMINAMATH_CALUDE_max_students_equal_distribution_l218_21834

theorem max_students_equal_distribution (pens pencils : ℕ) 
  (h_pens : pens = 640) (h_pencils : pencils = 520) :
  Nat.gcd pens pencils = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_students_equal_distribution_l218_21834


namespace NUMINAMATH_CALUDE_min_value_inequality_l218_21839

theorem min_value_inequality (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_eq : x + y + z = 9)
  (prod_sum_eq : x*y + y*z + z*x = 14) :
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l218_21839


namespace NUMINAMATH_CALUDE_correct_middle_managers_sample_l218_21836

/-- Represents the composition of employees in a company -/
structure CompanyComposition where
  total_employees : ℕ
  middle_managers : ℕ
  sample_size : ℕ

/-- Calculates the number of middle managers to be selected in a stratified random sample -/
def middleManagersInSample (comp : CompanyComposition) : ℕ :=
  (comp.sample_size * comp.middle_managers) / comp.total_employees

/-- Theorem stating that for the given company composition, 
    the number of middle managers in the sample should be 30 -/
theorem correct_middle_managers_sample :
  let comp : CompanyComposition := {
    total_employees := 1000,
    middle_managers := 150,
    sample_size := 200
  }
  middleManagersInSample comp = 30 := by
  sorry


end NUMINAMATH_CALUDE_correct_middle_managers_sample_l218_21836


namespace NUMINAMATH_CALUDE_ac_plus_bd_equals_negative_26_l218_21863

theorem ac_plus_bd_equals_negative_26
  (a b c d : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = 6) :
  a * c + b * d = -26 := by
  sorry

end NUMINAMATH_CALUDE_ac_plus_bd_equals_negative_26_l218_21863


namespace NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l218_21850

theorem cubic_polynomial_integer_root 
  (b c : ℚ) 
  (h1 : ∃ x : ℝ, x^3 + b*x + c = 0 ∧ x = 5 - Real.sqrt 11)
  (h2 : ∃ n : ℤ, (n : ℝ)^3 + b*(n : ℝ) + c = 0) :
  ∃ n : ℤ, (n : ℝ)^3 + b*(n : ℝ) + c = 0 ∧ n = -10 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l218_21850


namespace NUMINAMATH_CALUDE_constant_speed_motion_not_correlation_l218_21824

/-- Definition of a correlation relationship -/
def correlation_relationship (X Y : Type) (f : X → Y) :=
  ∃ (pattern : X → Set Y), ∀ x : X, f x ∈ pattern x ∧ ¬ (∃ y : Y, pattern x = {y})

/-- Definition of a functional relationship -/
def functional_relationship (X Y : Type) (f : X → Y) :=
  ∀ x : X, ∃! y : Y, f x = y

/-- Distance as a function of time for constant speed motion -/
def distance (v : ℝ) (t : ℝ) : ℝ := v * t

theorem constant_speed_motion_not_correlation :
  ∀ v : ℝ, v > 0 → ¬ (correlation_relationship ℝ ℝ (distance v)) :=
sorry

end NUMINAMATH_CALUDE_constant_speed_motion_not_correlation_l218_21824


namespace NUMINAMATH_CALUDE_f_inequality_l218_21865

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + a

-- State the theorem
theorem f_inequality (a b : ℝ) :
  (f a b 0 = 3) →
  (∀ x, f a b (2 - x) = f a b x) →
  (∀ x, f a b (b^x) ≤ f a b (a^x)) :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_l218_21865


namespace NUMINAMATH_CALUDE_unique_divisible_by_18_l218_21880

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem unique_divisible_by_18 :
  ∀ n : ℕ, n < 10 →
    (is_divisible_by (7120 + n) 18 ↔ n = 8) :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_by_18_l218_21880


namespace NUMINAMATH_CALUDE_hyperbola_equation_l218_21802

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if its eccentricity is 5/3 and the distance from its right focus to one asymptote is 4,
    then a = 3 and b = 4 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let e := 5/3  -- eccentricity
  let d := 4    -- distance from right focus to asymptote
  let c := Real.sqrt (a^2 + b^2)  -- distance from center to focus
  e = c/a ∧ d = b → a = 3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l218_21802


namespace NUMINAMATH_CALUDE_circle_circumference_approximation_l218_21898

/-- The circumference of a circle with radius 0.4997465213085514 meters is approximately 3.140093 meters. -/
theorem circle_circumference_approximation :
  let r : ℝ := 0.4997465213085514
  let π : ℝ := Real.pi
  let C : ℝ := 2 * π * r
  ∃ ε > 0, |C - 3.140093| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_approximation_l218_21898


namespace NUMINAMATH_CALUDE_quadratic_solution_l218_21835

theorem quadratic_solution (x : ℝ) : x^2 - 4*x + 3 = 0 → x = 1 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l218_21835


namespace NUMINAMATH_CALUDE_count_valid_arrangements_l218_21857

/-- The number of ways to arrange the digits 1, 1, 2, 5, 0 into a five-digit multiple of 5 -/
def arrangementCount : ℕ := 21

/-- The set of digits available for arrangement -/
def availableDigits : Finset ℕ := {1, 2, 5, 0}

/-- Predicate to check if a number is a five-digit multiple of 5 -/
def isValidNumber (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ n % 5 = 0

/-- The set of all valid arrangements -/
def validArrangements : Finset ℕ :=
  sorry

theorem count_valid_arrangements :
  Finset.card validArrangements = arrangementCount := by
  sorry

end NUMINAMATH_CALUDE_count_valid_arrangements_l218_21857


namespace NUMINAMATH_CALUDE_complement_of_angle_A_l218_21811

-- Define the angle A
def angle_A : ℝ := 76

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 90 - angle

-- Theorem statement
theorem complement_of_angle_A : complement angle_A = 14 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_angle_A_l218_21811


namespace NUMINAMATH_CALUDE_fifteen_factorial_base16_zeros_l218_21855

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to count trailing zeros in base 16
def trailingZerosBase16 (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

-- Theorem statement
theorem fifteen_factorial_base16_zeros :
  trailingZerosBase16 (factorial 15) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_factorial_base16_zeros_l218_21855


namespace NUMINAMATH_CALUDE_smallest_perimeter_l218_21894

/- Define the triangle PQR -/
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  isosceles : dist P Q = dist P R

/- Define the intersection point J -/
def J (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry

/- Define the perimeter of the triangle -/
def perimeter (P Q R : ℝ × ℝ) : ℝ :=
  dist P Q + dist Q R + dist R P

/- Main theorem -/
theorem smallest_perimeter (P Q R : ℝ × ℝ) (h : Triangle P Q R) :
  dist Q (J P Q R) = 10 →
  ∃ (P' Q' R' : ℝ × ℝ), Triangle P' Q' R' ∧
    dist Q' (J P' Q' R') = 10 ∧
    perimeter P' Q' R' = 198 ∧
    ∀ (P'' Q'' R'' : ℝ × ℝ), Triangle P'' Q'' R'' →
      dist Q'' (J P'' Q'' R'') = 10 →
      perimeter P'' Q'' R'' ≥ 198 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_l218_21894


namespace NUMINAMATH_CALUDE_sum_of_possible_base_3_digits_l218_21852

/-- The number of digits a positive integer has in a given base -/
def num_digits (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log base n + 1

/-- Checks if a number has exactly 4 digits in base 7 -/
def has_four_digits_base_7 (n : ℕ) : Prop :=
  num_digits n 7 = 4

/-- The smallest 4-digit number in base 7 -/
def min_four_digit_base_7 : ℕ := 7^3

/-- The largest 4-digit number in base 7 -/
def max_four_digit_base_7 : ℕ := 7^4 - 1

/-- The theorem to be proved -/
theorem sum_of_possible_base_3_digits : 
  (∀ n : ℕ, has_four_digits_base_7 n → 
    (num_digits n 3 = 6 ∨ num_digits n 3 = 7)) ∧ 
  (∃ n m : ℕ, has_four_digits_base_7 n ∧ has_four_digits_base_7 m ∧ 
    num_digits n 3 = 6 ∧ num_digits m 3 = 7) :=
sorry

end NUMINAMATH_CALUDE_sum_of_possible_base_3_digits_l218_21852


namespace NUMINAMATH_CALUDE_seashells_given_correct_l218_21899

/-- Calculates the number of seashells given away -/
def seashells_given (initial_seashells current_seashells : ℕ) : ℕ :=
  initial_seashells - current_seashells

/-- Proves that the number of seashells given away is correct -/
theorem seashells_given_correct (initial_seashells current_seashells : ℕ) 
  (h : initial_seashells ≥ current_seashells) :
  seashells_given initial_seashells current_seashells = initial_seashells - current_seashells :=
by
  sorry

#eval seashells_given 49 36

end NUMINAMATH_CALUDE_seashells_given_correct_l218_21899


namespace NUMINAMATH_CALUDE_equation_solution_set_l218_21840

theorem equation_solution_set (x : ℝ) : 
  (((9 : ℝ)^x + 32^x) / (15^x + 24^x) = 4/3) ↔ 
  (x = (Real.log (3/4)) / (Real.log (3/2)) ∨ x = (Real.log 4) / (Real.log 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_set_l218_21840


namespace NUMINAMATH_CALUDE_sum_of_roots_l218_21830

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 - 3*x^2 + 2000*x = 1997)
  (hy : y^3 - 3*y^2 + 2000*y = 1999) : 
  x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l218_21830


namespace NUMINAMATH_CALUDE_right_triangle_circle_chord_length_l218_21868

/-- Represents a triangle ABC with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem: In a right triangle ABC with hypotenuse AB = 10, AC = 8, and BC = 6,
    if a circle P passes through C and is tangent to AB at its midpoint,
    then the length of the chord QR (where Q and R are the intersections of
    the circle with AC and BC respectively) is equal to 9.6. -/
theorem right_triangle_circle_chord_length
  (abc : Triangle)
  (p : Circle)
  (h1 : abc.a = 10 ∧ abc.b = 8 ∧ abc.c = 6)
  (h2 : abc.a^2 = abc.b^2 + abc.c^2)
  (h3 : p.center = (5, p.radius))
  (h4 : p.radius = abc.b * abc.c / abc.a) :
  2 * p.radius = 9.6 := by sorry

end NUMINAMATH_CALUDE_right_triangle_circle_chord_length_l218_21868


namespace NUMINAMATH_CALUDE_find_number_l218_21866

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 75 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l218_21866


namespace NUMINAMATH_CALUDE_bird_watching_relationship_l218_21873

theorem bird_watching_relationship (cardinals robins blue_jays sparrows : ℕ) :
  cardinals = 3 →
  robins = 4 * cardinals →
  blue_jays = 2 * cardinals →
  cardinals + robins + blue_jays + sparrows = 31 →
  sparrows = (10 : ℚ) / 3 * cardinals :=
by sorry

end NUMINAMATH_CALUDE_bird_watching_relationship_l218_21873


namespace NUMINAMATH_CALUDE_turtle_path_max_entries_l218_21872

/-- Represents a turtle's path on a square grid -/
structure TurtlePath (n : ℕ) :=
  (grid_size : ℕ := 4*n + 2)
  (start_corner : Bool)
  (visits_all_squares_once : Bool)
  (ends_at_start : Bool)

/-- Represents the number of times a turtle enters a row or column -/
def max_entries (path : TurtlePath n) : ℕ := sorry

/-- Main theorem: There exists a row or column that the turtle enters at least 2n + 2 times -/
theorem turtle_path_max_entries {n : ℕ} (path : TurtlePath n) 
  (h1 : path.start_corner = true)
  (h2 : path.visits_all_squares_once = true)
  (h3 : path.ends_at_start = true) :
  max_entries path ≥ 2*n + 2 :=
sorry

end NUMINAMATH_CALUDE_turtle_path_max_entries_l218_21872


namespace NUMINAMATH_CALUDE_square_area_error_l218_21851

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := 1.08 * s
  let actual_area := s ^ 2
  let calculated_area := measured_side ^ 2
  let error_percentage := (calculated_area - actual_area) / actual_area * 100
  error_percentage = 16.64 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l218_21851


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l218_21820

theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 4 * x - 1 = 0) ↔ (a ≥ -4 ∧ a ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l218_21820


namespace NUMINAMATH_CALUDE_sphere_cross_section_distance_l218_21832

theorem sphere_cross_section_distance
  (V : ℝ) (A : ℝ) (d : ℝ)
  (hV : V = 4 * Real.sqrt 3 * Real.pi)
  (hA : A = Real.pi) :
  d = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_sphere_cross_section_distance_l218_21832


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l218_21827

/-- Given vectors a and b in ℝ², where a = (-1, 1) and b = (3, m),
    and a is parallel to (a + b), prove that m = -3. -/
theorem parallel_vectors_m_value (m : ℝ) : 
  let a : Fin 2 → ℝ := ![-1, 1]
  let b : Fin 2 → ℝ := ![3, m]
  (∃ (k : ℝ), k ≠ 0 ∧ (λ i => a i + b i) = λ i => k * a i) →
  m = -3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l218_21827


namespace NUMINAMATH_CALUDE_winnie_lollipop_distribution_l218_21823

theorem winnie_lollipop_distribution (total_lollipops : ℕ) (friends : ℕ) 
  (h1 : total_lollipops = 37 + 108 + 8 + 254) 
  (h2 : friends = 13) : 
  total_lollipops % friends = 4 := by
  sorry

end NUMINAMATH_CALUDE_winnie_lollipop_distribution_l218_21823


namespace NUMINAMATH_CALUDE_leah_lost_money_l218_21831

def total_earned : ℚ := 28
def milkshake_fraction : ℚ := 1/7
def savings_fraction : ℚ := 1/2
def remaining_in_wallet : ℚ := 1

theorem leah_lost_money : 
  let milkshake_cost := total_earned * milkshake_fraction
  let after_milkshake := total_earned - milkshake_cost
  let savings := after_milkshake * savings_fraction
  let in_wallet := after_milkshake - savings
  in_wallet - remaining_in_wallet = 11 := by sorry

end NUMINAMATH_CALUDE_leah_lost_money_l218_21831


namespace NUMINAMATH_CALUDE_salary_after_changes_l218_21805

-- Define the initial salary
def initial_salary : ℝ := 2000

-- Define the raise percentage
def raise_percentage : ℝ := 0.20

-- Define the pay cut percentage
def pay_cut_percentage : ℝ := 0.20

-- Theorem to prove
theorem salary_after_changes (s : ℝ) (r : ℝ) (c : ℝ) 
  (h1 : s = initial_salary) 
  (h2 : r = raise_percentage) 
  (h3 : c = pay_cut_percentage) : 
  s * (1 + r) * (1 - c) = 1920 := by
  sorry

end NUMINAMATH_CALUDE_salary_after_changes_l218_21805


namespace NUMINAMATH_CALUDE_inequality_proof_l218_21809

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_condition : a * b + b * c + c * d + d * a = 1) : 
  a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l218_21809


namespace NUMINAMATH_CALUDE_sum_of_cubes_inequality_l218_21825

theorem sum_of_cubes_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1)^3 / b + (b + 1)^3 / c + (c + 1)^3 / a ≥ 81 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_inequality_l218_21825


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l218_21849

/-- A positive arithmetic geometric sequence -/
def ArithGeomSeq (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The property that 3a_1, (1/2)a_3, 2a_2 form an arithmetic sequence -/
def ArithSeqProperty (a : ℕ → ℝ) : Prop :=
  (1/2) * a 3 = (3 * a 1 + 2 * a 2) / 2

theorem arithmetic_geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h1 : ArithGeomSeq a) 
  (h2 : ArithSeqProperty a) :
  (a 2016 - a 2017) / (a 2014 - a 2015) = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l218_21849


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l218_21896

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_point : (1^2 / a^2) + ((3/2)^2 / b^2) = 1
  h_ecc : (a^2 - b^2).sqrt / a = 1/2

/-- A line intersecting the ellipse -/
structure IntersectingLine (e : Ellipse) where
  k : ℝ
  m : ℝ
  h_k : k ≠ 0
  h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧
    (x₁^2 / 4) + (y₁^2 / 3) = 1 ∧
    (x₂^2 / 4) + (y₂^2 / 3) = 1 ∧
    y₁ = k * x₁ + m ∧
    y₂ = k * x₂ + m

/-- The main theorem -/
theorem ellipse_and_line_properties (e : Ellipse) (l : IntersectingLine e) :
  (∀ (x y : ℝ), (x^2 / 4) + (y^2 / 3) = 1 ↔ (x^2 / e.a^2) + (y^2 / e.b^2) = 1) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧
    (x₁^2 / 4) + (y₁^2 / 3) = 1 ∧
    (x₂^2 / 4) + (y₂^2 / 3) = 1 ∧
    y₁ = l.k * x₁ + l.m ∧
    y₂ = l.k * x₂ + l.m ∧
    (-(x₂ - x₁) * ((y₁ + y₂)/2 - 0) = (y₂ - y₁) * ((x₁ + x₂)/2 - 1/8)) →
    l.k < -Real.sqrt 5 / 10 ∨ l.k > Real.sqrt 5 / 10) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l218_21896


namespace NUMINAMATH_CALUDE_basketball_game_probability_formula_l218_21846

/-- Basketball shooting game between Student A and Student B -/
structure BasketballGame where
  /-- Probability of Student A making a basket -/
  prob_a : ℚ
  /-- Probability of Student B making a basket -/
  prob_b : ℚ
  /-- Each shot is independent -/
  independent_shots : Bool

/-- Score of Student A after one round -/
inductive Score where
  | lose : Score  -- Student A loses (-1)
  | draw : Score  -- Draw (0)
  | win  : Score  -- Student A wins (+1)

/-- Probability distribution of Student A's score after one round -/
def score_distribution (game : BasketballGame) : Score → ℚ
  | Score.lose => (1 - game.prob_a) * game.prob_b
  | Score.draw => game.prob_a * game.prob_b + (1 - game.prob_a) * (1 - game.prob_b)
  | Score.win  => game.prob_a * (1 - game.prob_b)

/-- Expected value of Student A's score after one round -/
def expected_score (game : BasketballGame) : ℚ :=
  -1 * score_distribution game Score.lose +
   0 * score_distribution game Score.draw +
   1 * score_distribution game Score.win

/-- Probability that Student A's cumulative score is lower than Student B's after n rounds -/
def p (n : ℕ) : ℚ :=
  (1 / 5) * (1 - (1 / 6)^n)

/-- Main theorem: Probability formula for Student A's score being lower after n rounds -/
theorem basketball_game_probability_formula (game : BasketballGame) (n : ℕ) 
    (h1 : game.prob_a = 2/3) (h2 : game.prob_b = 1/2) (h3 : game.independent_shots = true) :
    p n = (1 / 5) * (1 - (1 / 6)^n) := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_probability_formula_l218_21846


namespace NUMINAMATH_CALUDE_batsman_average_increase_l218_21842

/-- Represents a batsman's performance -/
structure Batsman where
  total_runs_before_16th : ℕ
  runs_in_16th : ℕ
  average_after_16th : ℚ

/-- Calculates the increase in average for a batsman -/
def average_increase (b : Batsman) : ℚ :=
  b.average_after_16th - (b.total_runs_before_16th : ℚ) / 15

/-- Theorem: The increase in average is 3 for a batsman who scores 64 runs in the 16th inning
    and has an average of 19 after the 16th inning -/
theorem batsman_average_increase
  (b : Batsman)
  (h1 : b.runs_in_16th = 64)
  (h2 : b.average_after_16th = 19)
  (h3 : b.total_runs_before_16th + b.runs_in_16th = 16 * b.average_after_16th) :
  average_increase b = 3 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_increase_l218_21842


namespace NUMINAMATH_CALUDE_job_completion_time_l218_21887

/-- Represents the time to complete a job given initial and final workforce conditions -/
def total_completion_time (n k t : ℕ) : ℝ :=
  t + 4 * (n + k)

/-- Theorem stating the total time to complete the job -/
theorem job_completion_time (n k t : ℕ) :
  (3 / 4 : ℝ) / t = n / total_completion_time n k t ∧
  (1 / 4 : ℝ) / (total_completion_time n k t - t) = (n + k) / (total_completion_time n k t - t) →
  total_completion_time n k t = t + 4 * (n + k) := by
  sorry

#check job_completion_time

end NUMINAMATH_CALUDE_job_completion_time_l218_21887


namespace NUMINAMATH_CALUDE_algorithmC_is_best_l218_21869

-- Define the durations of each task
def washAndBrush : ℕ := 5
def cleanKettle : ℕ := 2
def boilWater : ℕ := 8
def makeNoodles : ℕ := 3
def eat : ℕ := 10
def listenRadio : ℕ := 8

-- Define the algorithms
def algorithmA : ℕ := washAndBrush + cleanKettle + boilWater + makeNoodles + eat + listenRadio
def algorithmB : ℕ := cleanKettle + max boilWater washAndBrush + makeNoodles + eat + listenRadio
def algorithmC : ℕ := cleanKettle + max boilWater washAndBrush + makeNoodles + max eat listenRadio
def algorithmD : ℕ := max eat listenRadio + makeNoodles + max boilWater washAndBrush + cleanKettle

-- Theorem stating that algorithm C takes the least time
theorem algorithmC_is_best : 
  algorithmC ≤ algorithmA ∧ 
  algorithmC ≤ algorithmB ∧ 
  algorithmC ≤ algorithmD :=
sorry

end NUMINAMATH_CALUDE_algorithmC_is_best_l218_21869


namespace NUMINAMATH_CALUDE_regression_and_variance_l218_21853

-- Define the data points
def x : List Real := [5, 5.5, 6, 6.5, 7]
def y : List Real := [50, 48, 43, 38, 36]

-- Define the probability of "very good" experience
def p : Real := 0.5

-- Define the number of trials
def n : Nat := 5

-- Theorem statement
theorem regression_and_variance :
  let x_mean := (x.sum) / x.length
  let y_mean := (y.sum) / y.length
  let xy_sum := (List.zip x y).map (fun (a, b) => a * b) |>.sum
  let x_squared_sum := x.map (fun a => a ^ 2) |>.sum
  let slope := (xy_sum - x.length * x_mean * y_mean) / (x_squared_sum - x.length * x_mean ^ 2)
  let intercept := y_mean - slope * x_mean
  let variance := n * p * (1 - p)
  slope = -7.6 ∧ intercept = 88.6 ∧ variance = 5/4 := by
  sorry

#check regression_and_variance

end NUMINAMATH_CALUDE_regression_and_variance_l218_21853


namespace NUMINAMATH_CALUDE_unique_valid_number_l218_21801

def is_valid_number (abc : ℕ) : Prop :=
  let a := abc / 100
  let b := (abc / 10) % 10
  let c := abc % 10
  let sum := a + b + c
  abc % sum = 1 ∧
  (c * 100 + b * 10 + a) % sum = 1 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > c

theorem unique_valid_number : ∃! abc : ℕ, 100 ≤ abc ∧ abc < 1000 ∧ is_valid_number abc :=
  sorry

end NUMINAMATH_CALUDE_unique_valid_number_l218_21801


namespace NUMINAMATH_CALUDE_choose_four_from_twelve_l218_21856

theorem choose_four_from_twelve : Nat.choose 12 4 = 495 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_twelve_l218_21856


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l218_21891

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- The original point -/
def original_point : ℝ × ℝ := (-2, 3)

/-- The reflected point -/
def reflected_point : ℝ × ℝ := (2, 3)

theorem reflection_across_y_axis :
  reflect_y original_point = reflected_point := by sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l218_21891


namespace NUMINAMATH_CALUDE_original_price_calculation_l218_21848

/-- Given a 6% rebate followed by a 10% sales tax, if the final price is Rs. 6876.1,
    then the original price was Rs. 6650. -/
theorem original_price_calculation (original_price : ℝ) : 
  (original_price * (1 - 0.06) * (1 + 0.10) = 6876.1) → 
  (original_price = 6650) := by sorry

end NUMINAMATH_CALUDE_original_price_calculation_l218_21848


namespace NUMINAMATH_CALUDE_two_different_books_count_l218_21897

/-- The number of ways to select 2 books from different subjects -/
def selectTwoDifferentBooks (chinese : ℕ) (math : ℕ) (english : ℕ) : ℕ :=
  chinese * math + chinese * english + math * english

/-- Theorem: Given 9 Chinese books, 7 math books, and 5 English books,
    there are 143 ways to select 2 books from different subjects -/
theorem two_different_books_count :
  selectTwoDifferentBooks 9 7 5 = 143 := by
  sorry

end NUMINAMATH_CALUDE_two_different_books_count_l218_21897


namespace NUMINAMATH_CALUDE_equal_trout_division_l218_21803

theorem equal_trout_division (total_trout : ℕ) (num_people : ℕ) (trout_per_person : ℕ) : 
  total_trout = 18 → num_people = 2 → total_trout / num_people = trout_per_person → trout_per_person = 9 := by
  sorry

end NUMINAMATH_CALUDE_equal_trout_division_l218_21803


namespace NUMINAMATH_CALUDE_arithmetic_computation_l218_21871

theorem arithmetic_computation : -7 * 5 - (-4 * 3) + (-9 * -6) = 31 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l218_21871


namespace NUMINAMATH_CALUDE_yellow_balls_count_l218_21881

def total_balls : ℕ := 1500

def red_balls : ℕ := (2 * total_balls) / 7

def remaining_after_red : ℕ := total_balls - red_balls

def blue_balls : ℕ := (3 * remaining_after_red) / 11

def remaining_after_blue : ℕ := remaining_after_red - blue_balls

def green_balls : ℕ := remaining_after_blue / 5

def remaining_after_green : ℕ := remaining_after_blue - green_balls

def orange_balls : ℕ := remaining_after_green / 8

def yellow_balls : ℕ := remaining_after_green - orange_balls

theorem yellow_balls_count : yellow_balls = 546 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l218_21881


namespace NUMINAMATH_CALUDE_airplane_passengers_survey_is_census_l218_21816

/-- A survey type -/
inductive SurveyType
| FrozenFood
| AirplanePassengers
| RefrigeratorLifespan
| EnvironmentalAwareness

/-- Predicate for whether a survey requires examining every individual -/
def requiresExaminingAll (s : SurveyType) : Prop :=
  match s with
  | .AirplanePassengers => True
  | _ => False

/-- Definition of a census -/
def isCensus (s : SurveyType) : Prop :=
  requiresExaminingAll s

theorem airplane_passengers_survey_is_census :
  isCensus SurveyType.AirplanePassengers := by
  sorry

end NUMINAMATH_CALUDE_airplane_passengers_survey_is_census_l218_21816


namespace NUMINAMATH_CALUDE_maximum_of_sum_of_roots_l218_21884

theorem maximum_of_sum_of_roots (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 13) :
  Real.sqrt (x + 27) + Real.sqrt (13 - x) + Real.sqrt x ≤ 11 ∧
  ∃ y, 0 ≤ y ∧ y ≤ 13 ∧ Real.sqrt (y + 27) + Real.sqrt (13 - y) + Real.sqrt y = 11 :=
by sorry

end NUMINAMATH_CALUDE_maximum_of_sum_of_roots_l218_21884


namespace NUMINAMATH_CALUDE_smallest_three_digit_number_l218_21847

def Digits : Set Nat := {0, 3, 5, 6}

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100 ∈ Digits) ∧
  ((n / 10) % 10 ∈ Digits) ∧
  (n % 10 ∈ Digits) ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  (n / 100 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10)

theorem smallest_three_digit_number :
  ∀ n : Nat, is_valid_number n → n ≥ 305 :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_number_l218_21847


namespace NUMINAMATH_CALUDE_grade_students_count_l218_21874

theorem grade_students_count : ∃ n : ℕ, 
  400 < n ∧ n < 500 ∧
  n % 3 = 2 ∧
  n % 5 = 3 ∧
  n % 7 = 2 ∧
  n = 443 :=
by sorry

end NUMINAMATH_CALUDE_grade_students_count_l218_21874


namespace NUMINAMATH_CALUDE_phil_quarters_l218_21833

def initial_amount : ℚ := 40
def pizza_cost : ℚ := 2.75
def soda_cost : ℚ := 1.5
def jeans_cost : ℚ := 11.5

def remaining_amount : ℚ := initial_amount - (pizza_cost + soda_cost + jeans_cost)

def quarters_in_dollar : ℕ := 4

theorem phil_quarters : 
  ⌊remaining_amount * quarters_in_dollar⌋ = 97 := by
  sorry

end NUMINAMATH_CALUDE_phil_quarters_l218_21833


namespace NUMINAMATH_CALUDE_oomyapeck_eyes_eaten_l218_21885

/-- The number of eyes Oomyapeck eats given the family size, fish per person, eyes per fish, and eyes given away --/
def eyes_eaten (family_size : ℕ) (fish_per_person : ℕ) (eyes_per_fish : ℕ) (eyes_given_away : ℕ) : ℕ :=
  family_size * fish_per_person * eyes_per_fish - eyes_given_away

/-- Theorem stating that under the given conditions, Oomyapeck eats 22 eyes --/
theorem oomyapeck_eyes_eaten :
  eyes_eaten 3 4 2 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_oomyapeck_eyes_eaten_l218_21885


namespace NUMINAMATH_CALUDE_bookshelf_problem_l218_21893

/-- Represents the unit price of bookshelf type A -/
def price_A : ℕ := sorry

/-- Represents the unit price of bookshelf type B -/
def price_B : ℕ := sorry

/-- Represents the maximum number of type B bookshelves that can be purchased -/
def max_B : ℕ := sorry

theorem bookshelf_problem :
  (3 * price_A + 2 * price_B = 1020) ∧
  (price_A + 3 * price_B = 900) ∧
  (∀ m : ℕ, m ≤ 20 → price_A * (20 - m) + price_B * m ≤ 4350) →
  (price_A = 180 ∧ price_B = 240 ∧ max_B = 12) :=
by sorry

end NUMINAMATH_CALUDE_bookshelf_problem_l218_21893


namespace NUMINAMATH_CALUDE_truth_values_of_p_and_q_l218_21876

theorem truth_values_of_p_and_q (p q : Prop)
  (h1 : p ∨ q)
  (h2 : ¬(p ∧ q))
  (h3 : ¬p) :
  ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_truth_values_of_p_and_q_l218_21876
