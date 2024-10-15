import Mathlib

namespace NUMINAMATH_CALUDE_exponential_inequality_l2800_280055

theorem exponential_inequality (x y : ℝ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : Real.log x - Real.log y < 1 / Real.log x - 1 / Real.log y) : 
  Real.exp (y - x) > 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2800_280055


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l2800_280014

theorem line_intercepts_sum (d : ℚ) : 
  (∃ (x y : ℚ), 6 * x + 5 * y + d = 0 ∧ x + y = 15) → d = -450 / 11 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l2800_280014


namespace NUMINAMATH_CALUDE_right_triangle_area_l2800_280038

theorem right_triangle_area (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 72 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 216 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2800_280038


namespace NUMINAMATH_CALUDE_parabola_hyperbola_foci_coincide_l2800_280086

/-- The value of n for which the focus of the parabola y^2 = 8x coincides with 
    one of the foci of the hyperbola x^2/3 - y^2/n = 1 -/
theorem parabola_hyperbola_foci_coincide : ∃ n : ℝ,
  (∀ x y : ℝ, y^2 = 8*x → x^2/3 - y^2/n = 1) ∧ 
  (∃ x y : ℝ, y^2 = 8*x ∧ x^2/3 - y^2/n = 1 ∧ x = 2 ∧ y = 0) →
  n = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_foci_coincide_l2800_280086


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2800_280099

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  (X^5 - 1) * (X^3 - 1) = (X^3 + X^2 + 1) * q + (-2*X^2 + X + 1) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2800_280099


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_500_by_30_percent_l2800_280071

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) : 
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) :=
by sorry

theorem increase_500_by_30_percent : 
  500 * (1 + 30 / 100) = 650 :=
by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_500_by_30_percent_l2800_280071


namespace NUMINAMATH_CALUDE_james_writing_time_l2800_280040

/-- James' writing scenario -/
structure WritingScenario where
  pages_per_hour : ℕ
  pages_per_day_per_person : ℕ
  people_per_day : ℕ

/-- Calculate the hours spent writing per week -/
def hours_per_week (s : WritingScenario) : ℕ :=
  let pages_per_day := s.pages_per_day_per_person * s.people_per_day
  let pages_per_week := pages_per_day * 7
  pages_per_week / s.pages_per_hour

/-- Theorem: James spends 7 hours a week writing -/
theorem james_writing_time :
  let james := WritingScenario.mk 10 5 2
  hours_per_week james = 7 := by
  sorry

end NUMINAMATH_CALUDE_james_writing_time_l2800_280040


namespace NUMINAMATH_CALUDE_magnitude_squared_l2800_280046

theorem magnitude_squared (w : ℂ) (h : Complex.abs w = 11) : (2 * Complex.abs w)^2 = 484 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_squared_l2800_280046


namespace NUMINAMATH_CALUDE_sum_of_fractions_theorem_l2800_280003

variable (a b c P Q : ℝ)

theorem sum_of_fractions_theorem (h1 : a + b + c = 0) 
  (h2 : a^2 / (2*a^2 + b*c) + b^2 / (2*b^2 + a*c) + c^2 / (2*c^2 + a*b) = P - 3*Q) : 
  Q = 8 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_theorem_l2800_280003


namespace NUMINAMATH_CALUDE_prob_B_not_lose_l2800_280072

/-- The probability of player A winning in a chess game -/
def prob_A_win : ℝ := 0.3

/-- The probability of a draw in a chess game -/
def prob_draw : ℝ := 0.5

/-- Theorem: The probability of player B not losing in a chess game -/
theorem prob_B_not_lose : prob_A_win + prob_draw + (1 - prob_A_win - prob_draw) = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_prob_B_not_lose_l2800_280072


namespace NUMINAMATH_CALUDE_go_stones_theorem_l2800_280097

/-- Represents a stone on the grid -/
inductive Stone
| Black
| White

/-- Represents the grid configuration -/
def Grid (n : ℕ) := Fin (2*n) → Fin (2*n) → Option Stone

/-- Predicate to check if a stone exists at a given position -/
def has_stone (grid : Grid n) (i j : Fin (2*n)) : Prop :=
  ∃ (s : Stone), grid i j = some s

/-- Predicate to check if a black stone exists at a given position -/
def has_black_stone (grid : Grid n) (i j : Fin (2*n)) : Prop :=
  grid i j = some Stone.Black

/-- Predicate to check if a white stone exists at a given position -/
def has_white_stone (grid : Grid n) (i j : Fin (2*n)) : Prop :=
  grid i j = some Stone.White

/-- The grid after removing black stones that share a column with any white stone -/
def remove_black_stones (grid : Grid n) : Grid n :=
  sorry

/-- The grid after removing white stones that share a row with any remaining black stone -/
def remove_white_stones (grid : Grid n) : Grid n :=
  sorry

/-- Count the number of stones of a given type in the grid -/
def count_stones (grid : Grid n) (stone_type : Stone) : ℕ :=
  sorry

theorem go_stones_theorem (n : ℕ) (initial_grid : Grid n) :
  let final_grid := remove_white_stones (remove_black_stones initial_grid)
  (count_stones final_grid Stone.Black ≤ n^2) ∨ (count_stones final_grid Stone.White ≤ n^2) :=
sorry

end NUMINAMATH_CALUDE_go_stones_theorem_l2800_280097


namespace NUMINAMATH_CALUDE_parabola_focus_vertex_ratio_l2800_280031

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- The locus of midpoints of line segments AB on a parabola P where ∠AV₁B = 90° -/
def midpoint_locus (p : Parabola) : Parabola := sorry

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_focus_vertex_ratio :
  let p := Parabola.mk 4 0 0
  let q := midpoint_locus p
  let v1 := vertex p
  let v2 := vertex q
  let f1 := focus p
  let f2 := focus q
  distance f1 f2 / distance v1 v2 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_vertex_ratio_l2800_280031


namespace NUMINAMATH_CALUDE_parabola_chord_constant_l2800_280076

/-- Given a parabola y = 2x^2 and a point C(0, c), if t = 1/AC + 1/BC is constant
    for all chords AB passing through C, then t = -20/(7c) -/
theorem parabola_chord_constant (c : ℝ) :
  let parabola := fun (x : ℝ) => 2 * x^2
  let C := (0, c)
  let chord_length (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  ∃ t : ℝ, ∀ A B : ℝ × ℝ,
    A.2 = parabola A.1 →
    B.2 = parabola B.1 →
    (∃ m b : ℝ, ∀ x : ℝ, m * x + b = parabola x ↔ (x = A.1 ∨ x = B.1)) →
    C.2 = m * C.1 + b →
    t = 1 / chord_length A C + 1 / chord_length B C →
    t = -20 / (7 * c) :=
by
  sorry


end NUMINAMATH_CALUDE_parabola_chord_constant_l2800_280076


namespace NUMINAMATH_CALUDE_friday_to_monday_ratio_l2800_280090

def num_rabbits : ℕ := 16
def monday_toys : ℕ := 6
def wednesday_toys : ℕ := 2 * monday_toys
def saturday_toys : ℕ := wednesday_toys / 2
def toys_per_rabbit : ℕ := 3

def total_toys : ℕ := num_rabbits * toys_per_rabbit

def friday_toys : ℕ := total_toys - monday_toys - wednesday_toys - saturday_toys

theorem friday_to_monday_ratio :
  friday_toys / monday_toys = 4 ∧ friday_toys % monday_toys = 0 := by
  sorry

end NUMINAMATH_CALUDE_friday_to_monday_ratio_l2800_280090


namespace NUMINAMATH_CALUDE_sticker_distribution_l2800_280045

/-- The number of ways to distribute n identical objects among k distinct containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 8 identical stickers among 4 distinct sheets of paper -/
theorem sticker_distribution : distribute 8 4 = 165 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2800_280045


namespace NUMINAMATH_CALUDE_jesse_sam_earnings_l2800_280052

theorem jesse_sam_earnings (t : ℝ) : 
  t > 0 → 
  (t - 3) * (3 * t - 4) = 2 * (3 * t - 6) * (t - 3) → 
  t = 4 := by
sorry

end NUMINAMATH_CALUDE_jesse_sam_earnings_l2800_280052


namespace NUMINAMATH_CALUDE_sqrt_eight_times_sqrt_two_l2800_280008

theorem sqrt_eight_times_sqrt_two : Real.sqrt 8 * Real.sqrt 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_times_sqrt_two_l2800_280008


namespace NUMINAMATH_CALUDE_equation_solution_l2800_280033

theorem equation_solution : ∃! x : ℝ, (x + 1 ≠ 0 ∧ 2*x - 1 ≠ 0) ∧ (2 / (x + 1) = 3 / (2*x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2800_280033


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2800_280088

/-- Given a point A(a, 4) in the second quadrant and a vertical line m with x = 2,
    the point symmetric to A with respect to m has coordinates (4-a, 4). -/
theorem symmetric_point_coordinates (a : ℝ) (h1 : a < 0) :
  let A : ℝ × ℝ := (a, 4)
  let m : Set (ℝ × ℝ) := {p | p.1 = 2}
  let symmetric_point := (4 - a, 4)
  symmetric_point.1 = 4 - a ∧ symmetric_point.2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2800_280088


namespace NUMINAMATH_CALUDE_distance_FM_l2800_280093

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 1)

-- Define the length of AB
def length_AB : ℝ := 6

-- Define the perpendicular bisector of AB
def perp_bisector (k : ℝ) (x y : ℝ) : Prop :=
  y - k = -(1/k) * (x - 2)

-- Define the point M
def point_M (k : ℝ) : ℝ × ℝ :=
  (4, 0)

-- Theorem statement
theorem distance_FM (k : ℝ) :
  let F := focus
  let M := point_M k
  (M.1 - F.1)^2 + (M.2 - F.2)^2 = 3^2 :=
sorry

end NUMINAMATH_CALUDE_distance_FM_l2800_280093


namespace NUMINAMATH_CALUDE_scout_profit_is_250_l2800_280007

/-- Calculates the profit for a scout troop selling candy bars -/
def scout_profit (num_bars : ℕ) (buy_rate : ℚ) (sell_rate : ℚ) : ℚ :=
  let cost_per_bar := 3 / (6 : ℚ)
  let sell_per_bar := 2 / (3 : ℚ)
  let total_cost := (num_bars : ℚ) * cost_per_bar
  let total_revenue := (num_bars : ℚ) * sell_per_bar
  total_revenue - total_cost

/-- The profit for a scout troop selling 1500 candy bars is $250 -/
theorem scout_profit_is_250 :
  scout_profit 1500 (3/6) (2/3) = 250 := by
  sorry

end NUMINAMATH_CALUDE_scout_profit_is_250_l2800_280007


namespace NUMINAMATH_CALUDE_profit_share_ratio_l2800_280036

/-- Given two investors P and Q with their respective investments, 
    calculate the ratio of their profit shares. -/
theorem profit_share_ratio 
  (p_investment q_investment : ℕ) 
  (h_p : p_investment = 40000) 
  (h_q : q_investment = 60000) : 
  (p_investment : ℚ) / q_investment = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_ratio_l2800_280036


namespace NUMINAMATH_CALUDE_max_value_of_x_l2800_280044

theorem max_value_of_x (x y z : ℝ) 
  (sum_eq : x + y + z = 3) 
  (prod_sum_eq : x * y + x * z + y * z = 3) : 
  x ≤ 1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_x_l2800_280044


namespace NUMINAMATH_CALUDE_pauline_snow_shoveling_l2800_280015

/-- Calculates the total volume of snow shoveled up to a given hour -/
def snowShoveled (hour : ℕ) : ℕ :=
  (20 * hour) - (hour * (hour - 1) / 2)

/-- Represents Pauline's snow shoveling problem -/
theorem pauline_snow_shoveling (drivewayWidth drivewayLength snowDepth : ℕ) 
  (h1 : drivewayWidth = 5)
  (h2 : drivewayLength = 10)
  (h3 : snowDepth = 4) :
  ∃ (hour : ℕ), hour = 13 ∧ snowShoveled hour ≥ drivewayWidth * drivewayLength * snowDepth ∧ 
  snowShoveled (hour - 1) < drivewayWidth * drivewayLength * snowDepth :=
by
  sorry


end NUMINAMATH_CALUDE_pauline_snow_shoveling_l2800_280015


namespace NUMINAMATH_CALUDE_greatest_value_b_l2800_280057

theorem greatest_value_b (b : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + 3 < -x + 6 → x ≤ b) ↔ b = (3 + Real.sqrt 21) / 2 :=
sorry

end NUMINAMATH_CALUDE_greatest_value_b_l2800_280057


namespace NUMINAMATH_CALUDE_rook_paths_on_chessboard_l2800_280095

def rook_paths (n m k : ℕ) : ℕ :=
  if n + m ≠ k then 0
  else Nat.choose (n + m) n

theorem rook_paths_on_chessboard :
  (rook_paths 7 7 14 = 3432) ∧
  (rook_paths 7 7 12 = 57024) ∧
  (rook_paths 7 7 5 = 2000) := by
  sorry

end NUMINAMATH_CALUDE_rook_paths_on_chessboard_l2800_280095


namespace NUMINAMATH_CALUDE_mean_of_smallest_elements_l2800_280094

/-- F(n, r) represents the arithmetic mean of the smallest elements in all r-element subsets of {1, 2, ..., n} -/
def F (n r : ℕ) : ℚ :=
  sorry

/-- Theorem stating that F(n, r) = (n+1)/(r+1) for 1 ≤ r ≤ n -/
theorem mean_of_smallest_elements (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) :
  F n r = (n + 1 : ℚ) / (r + 1 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_mean_of_smallest_elements_l2800_280094


namespace NUMINAMATH_CALUDE_tangent_line_circle_min_value_l2800_280002

theorem tangent_line_circle_min_value (a b : ℝ) :
  a > 0 →
  b > 0 →
  a^2 + 4*b^2 = 2 →
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a*x + 2*b*y + 2 = 0 ∧ x^2 + y^2 = 2) →
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → a'^2 + 4*b'^2 = 2 →
    (∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ a'*x' + 2*b'*y' + 2 = 0 ∧ x'^2 + y'^2 = 2) →
    1/a^2 + 1/b^2 ≤ 1/a'^2 + 1/b'^2) →
  1/a^2 + 1/b^2 = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_min_value_l2800_280002


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2800_280012

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2800_280012


namespace NUMINAMATH_CALUDE_area_between_parabola_and_line_l2800_280039

theorem area_between_parabola_and_line : 
  let f (x : ℝ) := x^2
  let g (x : ℝ) := x
  let area := ∫ x in (0:ℝ)..1, (g x - f x)
  area = 1/6 := by sorry

end NUMINAMATH_CALUDE_area_between_parabola_and_line_l2800_280039


namespace NUMINAMATH_CALUDE_anya_lost_games_l2800_280091

/-- Represents a girl in the table tennis game --/
inductive Girl
| Anya
| Bella
| Valya
| Galya
| Dasha

/-- Represents a game of table tennis --/
structure Game where
  number : Nat
  players : Fin 2 → Girl
  loser : Girl

/-- The total number of games played --/
def total_games : Nat := 19

/-- The number of games each girl played --/
def games_played (g : Girl) : Nat :=
  match g with
  | Girl.Anya => 4
  | Girl.Bella => 6
  | Girl.Valya => 7
  | Girl.Galya => 10
  | Girl.Dasha => 11

/-- Predicate to check if a girl lost a specific game --/
def lost_game (g : Girl) (n : Nat) : Prop := ∃ game : Game, game.number = n ∧ game.loser = g

/-- The main theorem to prove --/
theorem anya_lost_games :
  (lost_game Girl.Anya 4) ∧
  (lost_game Girl.Anya 8) ∧
  (lost_game Girl.Anya 12) ∧
  (lost_game Girl.Anya 16) ∧
  (∀ n : Nat, n ≤ total_games → n ≠ 4 → n ≠ 8 → n ≠ 12 → n ≠ 16 → ¬(lost_game Girl.Anya n)) :=
sorry

end NUMINAMATH_CALUDE_anya_lost_games_l2800_280091


namespace NUMINAMATH_CALUDE_line_y_axis_intersection_l2800_280001

/-- The line equation is 5y + 3x = 15 -/
def line_equation (x y : ℝ) : Prop := 5 * y + 3 * x = 15

/-- A point is on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- Theorem: The point (0, 3) is the intersection of the line 5y + 3x = 15 with the y-axis -/
theorem line_y_axis_intersection :
  line_equation 0 3 ∧ on_y_axis 0 3 ∧
  ∀ x y : ℝ, line_equation x y ∧ on_y_axis x y → x = 0 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_y_axis_intersection_l2800_280001


namespace NUMINAMATH_CALUDE_line_equation_l2800_280060

/-- A line passing through point A(1,4) with the sum of its intercepts on the two axes equal to zero -/
structure LineWithZeroSumIntercepts where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through point A(1,4) -/
  passes_through_A : 4 = slope * 1 + y_intercept
  /-- The sum of intercepts on the two axes is zero -/
  zero_sum_intercepts : (-y_intercept / slope) + y_intercept = 0

/-- The equation of the line is either 4x - y = 0 or x - y + 3 = 0 -/
theorem line_equation (l : LineWithZeroSumIntercepts) :
  (l.slope = 4 ∧ l.y_intercept = 0) ∨ (l.slope = 1 ∧ l.y_intercept = 3) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l2800_280060


namespace NUMINAMATH_CALUDE_paint_cube_cost_l2800_280077

/-- The cost to paint a cube given paint cost, coverage, and cube dimensions -/
theorem paint_cube_cost 
  (paint_cost : ℝ)       -- Cost of paint per kg in Rs
  (paint_coverage : ℝ)   -- Area covered by 1 kg of paint in sq. ft
  (cube_side : ℝ)        -- Length of cube side in feet
  (h1 : paint_cost = 20) -- Paint costs 20 Rs per kg
  (h2 : paint_coverage = 15) -- 1 kg of paint covers 15 sq. ft
  (h3 : cube_side = 5)   -- Cube side is 5 feet
  : ℝ :=
by
  -- The cost to paint the cube is 200 Rs
  sorry

#check paint_cube_cost

end NUMINAMATH_CALUDE_paint_cube_cost_l2800_280077


namespace NUMINAMATH_CALUDE_no_thirty_consecutive_zeros_l2800_280019

/-- For any natural number n, the last 100 digits of 5^n do not contain 30 consecutive zeros. -/
theorem no_thirty_consecutive_zeros (n : ℕ) : 
  ¬ (∃ k : ℕ, k + 29 < 100 ∧ ∀ i : ℕ, i < 30 → (5^n / 10^k) % 10^(100-k) % 10 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_thirty_consecutive_zeros_l2800_280019


namespace NUMINAMATH_CALUDE_problem_statement_l2800_280023

theorem problem_statement (a b : ℝ) : 
  (Real.sqrt (a - 2) + abs (b + 3) = 0) → ((a + b) ^ 2023 = -1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2800_280023


namespace NUMINAMATH_CALUDE_chess_grandmaster_learning_time_l2800_280092

theorem chess_grandmaster_learning_time 
  (total_time : ℕ) 
  (proficiency_multiplier : ℕ) 
  (mastery_multiplier : ℕ) 
  (h1 : total_time = 10100)
  (h2 : proficiency_multiplier = 49)
  (h3 : mastery_multiplier = 100) : 
  ∃ (rule_learning_time : ℕ), 
    rule_learning_time = 2 ∧ 
    total_time = rule_learning_time + 
                 proficiency_multiplier * rule_learning_time + 
                 mastery_multiplier * (rule_learning_time + proficiency_multiplier * rule_learning_time) :=
by sorry

end NUMINAMATH_CALUDE_chess_grandmaster_learning_time_l2800_280092


namespace NUMINAMATH_CALUDE_expression_evaluation_l2800_280043

theorem expression_evaluation (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  let x := (4 * a * b) / (a + b)
  ((x + 2*b) / (x - 2*b) + (x + 2*a) / (x - 2*a)) / (x / 2) = (a + b) / (a * b) :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2800_280043


namespace NUMINAMATH_CALUDE_factorization_1_l2800_280058

theorem factorization_1 (m n : ℝ) :
  3 * m^2 * n - 12 * m * n + 12 * n = 3 * n * (m - 2)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_1_l2800_280058


namespace NUMINAMATH_CALUDE_price_reduction_for_target_profit_max_profit_price_reduction_l2800_280009

/-- Profit function given price reduction x -/
def profit (x : ℝ) : ℝ := (80 - x) * (40 + 2 * x)

/-- Theorem for part 1 of the problem -/
theorem price_reduction_for_target_profit :
  profit 40 = 4800 := by sorry

/-- Theorem for part 2 of the problem -/
theorem max_profit_price_reduction :
  ∀ x : ℝ, profit x ≤ profit 30 ∧ profit 30 = 5000 := by sorry

end NUMINAMATH_CALUDE_price_reduction_for_target_profit_max_profit_price_reduction_l2800_280009


namespace NUMINAMATH_CALUDE_essay_competition_probability_l2800_280037

theorem essay_competition_probability (n : ℕ) (h : n = 6) :
  let p := (n - 1) / n
  p = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_essay_competition_probability_l2800_280037


namespace NUMINAMATH_CALUDE_min_slide_time_l2800_280062

/-- A vertical circle fixed to a horizontal line -/
structure VerticalCircle where
  center : ℝ × ℝ
  radius : ℝ
  is_vertical : center.2 = radius

/-- A point outside and above the circle -/
structure OutsidePoint (C : VerticalCircle) where
  coords : ℝ × ℝ
  is_outside : (coords.1 - C.center.1)^2 + (coords.2 - C.center.2)^2 > C.radius^2
  is_above : coords.2 > C.center.2 + C.radius

/-- A point on the circle -/
def CirclePoint (C : VerticalCircle) := { p : ℝ × ℝ // (p.1 - C.center.1)^2 + (p.2 - C.center.2)^2 = C.radius^2 }

/-- The time function for a particle to slide down from P to Q under gravity -/
noncomputable def slide_time (C : VerticalCircle) (P : OutsidePoint C) (Q : CirclePoint C) : ℝ := sorry

/-- The lowest point on the circle -/
def lowest_point (C : VerticalCircle) : CirclePoint C :=
  ⟨(C.center.1, C.center.2 - C.radius), sorry⟩

/-- Theorem: The point Q that minimizes the slide time is the lowest point on the circle -/
theorem min_slide_time (C : VerticalCircle) (P : OutsidePoint C) :
  ∀ Q : CirclePoint C, slide_time C P Q ≥ slide_time C P (lowest_point C) :=
sorry

end NUMINAMATH_CALUDE_min_slide_time_l2800_280062


namespace NUMINAMATH_CALUDE_total_leaves_on_ferns_l2800_280084

theorem total_leaves_on_ferns : 
  let total_ferns : ℕ := 12
  let type_a_ferns : ℕ := 4
  let type_b_ferns : ℕ := 5
  let type_c_ferns : ℕ := 3
  let type_a_fronds : ℕ := 15
  let type_a_leaves_per_frond : ℕ := 45
  let type_b_fronds : ℕ := 20
  let type_b_leaves_per_frond : ℕ := 30
  let type_c_fronds : ℕ := 25
  let type_c_leaves_per_frond : ℕ := 40

  total_ferns = type_a_ferns + type_b_ferns + type_c_ferns →
  (type_a_ferns * type_a_fronds * type_a_leaves_per_frond +
   type_b_ferns * type_b_fronds * type_b_leaves_per_frond +
   type_c_ferns * type_c_fronds * type_c_leaves_per_frond) = 8700 :=
by
  sorry

#check total_leaves_on_ferns

end NUMINAMATH_CALUDE_total_leaves_on_ferns_l2800_280084


namespace NUMINAMATH_CALUDE_factor_divisor_statements_l2800_280080

theorem factor_divisor_statements : 
  (∃ n : ℤ, 24 = 4 * n) ∧ 
  (∃ n : ℤ, 209 = 19 * n) ∧ 
  ¬(∃ n : ℤ, 63 = 19 * n) ∧
  (∃ n : ℤ, 180 = 9 * n) := by
sorry

end NUMINAMATH_CALUDE_factor_divisor_statements_l2800_280080


namespace NUMINAMATH_CALUDE_shaded_squares_area_sum_square_division_problem_l2800_280018

theorem shaded_squares_area_sum (initial_area : ℝ) (ratio : ℝ) :
  initial_area > 0 →
  ratio > 0 →
  ratio < 1 →
  let series_sum := initial_area / (1 - ratio)
  series_sum = initial_area / (1 - ratio) :=
by sorry

theorem square_division_problem :
  let initial_side_length : ℝ := 8
  let initial_area : ℝ := initial_side_length ^ 2
  let ratio : ℝ := 1 / 4
  let series_sum := initial_area / (1 - ratio)
  series_sum = 64 / 3 :=
by sorry

end NUMINAMATH_CALUDE_shaded_squares_area_sum_square_division_problem_l2800_280018


namespace NUMINAMATH_CALUDE_complex_magnitude_l2800_280069

theorem complex_magnitude (z w : ℂ) 
  (h1 : Complex.abs (2 * z - w) = 25)
  (h2 : Complex.abs (z + 2 * w) = 5)
  (h3 : Complex.abs (z + w) = 2) : 
  Complex.abs z = 9 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2800_280069


namespace NUMINAMATH_CALUDE_original_people_in_room_l2800_280011

theorem original_people_in_room (x : ℝ) : 
  (x / 2 = 15) → 
  (x / 3 + x / 4 * (2 / 3) + 15 = x) → 
  x = 30 := by
sorry

end NUMINAMATH_CALUDE_original_people_in_room_l2800_280011


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_2sqrt3_l2800_280022

theorem sqrt_difference_equals_2sqrt3 : 
  Real.sqrt (7 + 4 * Real.sqrt 3) - Real.sqrt (7 - 4 * Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_2sqrt3_l2800_280022


namespace NUMINAMATH_CALUDE_perfect_square_sum_l2800_280025

theorem perfect_square_sum (x y : ℕ) 
  (h : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / (x + 2) + (1 : ℚ) / (y - 2)) : 
  ∃ n : ℕ, x * y + 1 = n ^ 2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l2800_280025


namespace NUMINAMATH_CALUDE_product_of_roots_l2800_280032

theorem product_of_roots (a b c d : ℂ) : 
  (3 * a^4 - 8 * a^3 + a^2 + 4 * a - 10 = 0) ∧ 
  (3 * b^4 - 8 * b^3 + b^2 + 4 * b - 10 = 0) ∧ 
  (3 * c^4 - 8 * c^3 + c^2 + 4 * c - 10 = 0) ∧ 
  (3 * d^4 - 8 * d^3 + d^2 + 4 * d - 10 = 0) →
  a * b * c * d = -10/3 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l2800_280032


namespace NUMINAMATH_CALUDE_sqrt_sum_squared_l2800_280054

theorem sqrt_sum_squared (x y z : ℝ) : 
  (Real.sqrt 80 + 3 * Real.sqrt 5 + Real.sqrt 450 / 3)^2 = 295 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squared_l2800_280054


namespace NUMINAMATH_CALUDE_expression_evaluation_l2800_280029

theorem expression_evaluation : 2^3 + 4 * 5 - Real.sqrt 9 + (3^2 * 2) / 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2800_280029


namespace NUMINAMATH_CALUDE_inverse_function_point_correspondence_l2800_280041

theorem inverse_function_point_correspondence 
  (f : ℝ → ℝ) (hf : Function.Bijective f) :
  (1 - f 1 = 2) → (f⁻¹ (-1) - (-1) = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_function_point_correspondence_l2800_280041


namespace NUMINAMATH_CALUDE_student_committee_size_l2800_280061

theorem student_committee_size (ways_to_select : ℕ) (h : ways_to_select = 42) :
  ∃ n : ℕ, n > 1 ∧ n * (n - 1) = ways_to_select ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_student_committee_size_l2800_280061


namespace NUMINAMATH_CALUDE_rent_expenditure_l2800_280026

def monthly_salary : ℕ := 18000
def savings_percentage : ℚ := 1/10
def savings : ℕ := 1800
def milk_expense : ℕ := 1500
def groceries_expense : ℕ := 4500
def education_expense : ℕ := 2500
def petrol_expense : ℕ := 2000
def misc_expense : ℕ := 700

theorem rent_expenditure :
  let total_expenses := milk_expense + groceries_expense + education_expense + petrol_expense + misc_expense
  let rent := monthly_salary - (total_expenses + savings)
  (savings = (savings_percentage * monthly_salary).num) →
  rent = 6000 := by sorry

end NUMINAMATH_CALUDE_rent_expenditure_l2800_280026


namespace NUMINAMATH_CALUDE_min_difference_of_product_l2800_280051

theorem min_difference_of_product (a b : ℤ) (h : a * b = 156) :
  ∀ x y : ℤ, x * y = 156 → a - b ≤ x - y :=
by sorry

end NUMINAMATH_CALUDE_min_difference_of_product_l2800_280051


namespace NUMINAMATH_CALUDE_jim_gave_away_900_cards_l2800_280096

/-- The number of cards Jim gave away -/
def cards_given_away (initial_cards : ℕ) (set_size : ℕ) (sets_to_brother sets_to_sister sets_to_friend sets_to_cousin sets_to_classmate : ℕ) : ℕ :=
  (sets_to_brother + sets_to_sister + sets_to_friend + sets_to_cousin + sets_to_classmate) * set_size

/-- Proof that Jim gave away 900 cards -/
theorem jim_gave_away_900_cards :
  cards_given_away 1500 25 15 8 4 6 3 = 900 := by
  sorry

end NUMINAMATH_CALUDE_jim_gave_away_900_cards_l2800_280096


namespace NUMINAMATH_CALUDE_product_97_squared_l2800_280035

theorem product_97_squared : 97 * 97 = 9409 := by
  sorry

end NUMINAMATH_CALUDE_product_97_squared_l2800_280035


namespace NUMINAMATH_CALUDE_square_area_proof_l2800_280063

theorem square_area_proof (s : ℝ) (h1 : s = 4) : 
  (s^2 + s) - (4 * s) = 4 → s^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_square_area_proof_l2800_280063


namespace NUMINAMATH_CALUDE_divisible_by_4_or_6_count_l2800_280066

def countDivisibleByEither (n : ℕ) (a b : ℕ) : ℕ :=
  (n / a) + (n / b) - (n / (Nat.lcm a b))

theorem divisible_by_4_or_6_count :
  countDivisibleByEither 80 4 6 = 27 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_4_or_6_count_l2800_280066


namespace NUMINAMATH_CALUDE_equidistant_chord_length_l2800_280089

theorem equidistant_chord_length 
  (d : ℝ) 
  (c1 c2 : ℝ) 
  (dist : ℝ) 
  (h1 : d = 20) 
  (h2 : c1 = 10) 
  (h3 : c2 = 14) 
  (h4 : dist = 6) :
  ∃ (x : ℝ), x^2 = 164 ∧ 
  (∃ (y : ℝ), y > 0 ∧ y < dist ∧
    (d/2)^2 = (c1/2)^2 + y^2 ∧
    (d/2)^2 = (c2/2)^2 + (dist - y)^2 ∧
    x^2/4 + (y + (dist - y)/2)^2 = (d/2)^2) :=
by sorry

end NUMINAMATH_CALUDE_equidistant_chord_length_l2800_280089


namespace NUMINAMATH_CALUDE_area_triangle_ADG_l2800_280047

/-- Regular octagon with side length 3 -/
structure RegularOctagon where
  side_length : ℝ
  is_regular : side_length = 3

/-- Triangle ADG in the regular octagon -/
def TriangleADG (octagon : RegularOctagon) : Set (Fin 3 → ℝ × ℝ) :=
  sorry

/-- Area of a triangle -/
def triangleArea (triangle : Set (Fin 3 → ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: Area of triangle ADG in a regular octagon with side length 3 -/
theorem area_triangle_ADG (octagon : RegularOctagon) :
  triangleArea (TriangleADG octagon) = (27 - 9 * Real.sqrt 2 + 9 * Real.sqrt (2 - 2 * Real.sqrt 2)) / (2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_ADG_l2800_280047


namespace NUMINAMATH_CALUDE_bea_lemonade_sales_l2800_280004

theorem bea_lemonade_sales (bea_price dawn_price : ℚ) (dawn_sales : ℕ) (extra_earnings : ℚ) :
  bea_price = 25/100 →
  dawn_price = 28/100 →
  dawn_sales = 8 →
  extra_earnings = 26/100 →
  ∃ bea_sales : ℕ, bea_sales * bea_price = dawn_sales * dawn_price + extra_earnings ∧ bea_sales = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_bea_lemonade_sales_l2800_280004


namespace NUMINAMATH_CALUDE_committee_with_female_count_l2800_280005

def total_members : ℕ := 30
def female_members : ℕ := 12
def male_members : ℕ := 18
def committee_size : ℕ := 5

theorem committee_with_female_count :
  (Nat.choose total_members committee_size) - (Nat.choose male_members committee_size) = 133938 :=
by sorry

end NUMINAMATH_CALUDE_committee_with_female_count_l2800_280005


namespace NUMINAMATH_CALUDE_min_value_of_a_l2800_280083

theorem min_value_of_a (a : ℝ) (h1 : a > 0) : 
  (∀ x : ℝ, x > 1 → x + a / (x - 1) ≥ 5) → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l2800_280083


namespace NUMINAMATH_CALUDE_evaluate_expression_l2800_280027

theorem evaluate_expression (x : ℝ) (h : x = 6) : 
  (x^9 - 24*x^6 + 144*x^3 - 512) / (x^3 - 8) = 43264 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2800_280027


namespace NUMINAMATH_CALUDE_cube_sum_from_sixth_power_sum_l2800_280085

theorem cube_sum_from_sixth_power_sum (x : ℝ) (h : 47 = x^6 + 1/x^6) : x^3 + 1/x^3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_from_sixth_power_sum_l2800_280085


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2800_280074

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2800_280074


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2800_280021

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a > 0 → a^2 + a ≥ 0) ∧ ¬(a^2 + a ≥ 0 → a > 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2800_280021


namespace NUMINAMATH_CALUDE_min_value_of_sequence_l2800_280048

theorem min_value_of_sequence (n : ℝ) : 
  ∃ (m : ℝ), ∀ (n : ℝ), n^2 - 8*n + 5 ≥ m ∧ ∃ (k : ℝ), k^2 - 8*k + 5 = m :=
by
  -- The minimum value is -11
  use -11
  sorry

end NUMINAMATH_CALUDE_min_value_of_sequence_l2800_280048


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2800_280013

theorem quadratic_inequality_range (α : Real) (h : 0 ≤ α ∧ α ≤ π) :
  (∀ x : Real, 8 * x^2 - (8 * Real.sin α) * x + Real.cos (2 * α) ≥ 0) ↔
  (0 ≤ α ∧ α ≤ π / 6) ∨ (5 * π / 6 ≤ α ∧ α ≤ π) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2800_280013


namespace NUMINAMATH_CALUDE_ternary_35_implies_k_2_l2800_280053

def ternary_to_decimal (k : ℕ+) : ℕ := 1 * 3^3 + k * 3^2 + 2

theorem ternary_35_implies_k_2 : 
  ∀ k : ℕ+, ternary_to_decimal k = 35 → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ternary_35_implies_k_2_l2800_280053


namespace NUMINAMATH_CALUDE_peasant_money_problem_l2800_280098

theorem peasant_money_problem (initial_money : ℕ) : 
  let after_first := initial_money / 2 - 1
  let after_second := after_first / 2 - 2
  let after_third := after_second / 2 - 1
  (after_third = 0) → initial_money = 6 := by
sorry

end NUMINAMATH_CALUDE_peasant_money_problem_l2800_280098


namespace NUMINAMATH_CALUDE_hash_composition_l2800_280042

-- Define the # operation
def hash (a b : ℝ) : ℝ := a * b - b + b^2

-- Theorem statement
theorem hash_composition (z : ℝ) : hash (hash 3 8) z = 79 * z + z^2 := by
  sorry

end NUMINAMATH_CALUDE_hash_composition_l2800_280042


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2800_280017

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2800_280017


namespace NUMINAMATH_CALUDE_a_minus_b_is_perfect_square_l2800_280064

theorem a_minus_b_is_perfect_square (a b : ℕ+) (h : 2 * a ^ 2 + a = 3 * b ^ 2 + b) :
  ∃ k : ℕ, (a : ℤ) - (b : ℤ) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_is_perfect_square_l2800_280064


namespace NUMINAMATH_CALUDE_exists_convex_polyhedron_with_triangular_section_l2800_280049

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  -- This is a placeholder structure

/-- A cross-section of a polyhedron -/
structure CrossSection where
  -- Add necessary fields here
  -- This is a placeholder structure

/-- Predicate to check if a cross-section is triangular -/
def is_triangular (cs : CrossSection) : Prop :=
  sorry

/-- Predicate to check if a cross-section passes through vertices -/
def passes_through_vertices (p : ConvexPolyhedron) (cs : CrossSection) : Prop :=
  sorry

/-- Function to count the number of edges meeting at a vertex -/
def edges_at_vertex (p : ConvexPolyhedron) (v : ℕ) : ℕ :=
  sorry

/-- Theorem stating the existence of a convex polyhedron with the specified properties -/
theorem exists_convex_polyhedron_with_triangular_section :
  ∃ (p : ConvexPolyhedron) (cs : CrossSection),
    is_triangular cs ∧
    ¬passes_through_vertices p cs ∧
    ∀ (v : ℕ), edges_at_vertex p v = 5 :=
  sorry

end NUMINAMATH_CALUDE_exists_convex_polyhedron_with_triangular_section_l2800_280049


namespace NUMINAMATH_CALUDE_angle_from_point_l2800_280068

theorem angle_from_point (θ : Real) (h1 : θ ∈ Set.Icc 0 (2 * Real.pi)) :
  (∃ (P : ℝ × ℝ), P.1 = Real.sin (3 * Real.pi / 4) ∧ 
                   P.2 = Real.cos (3 * Real.pi / 4) ∧ 
                   P.1 = Real.sin θ ∧ 
                   P.2 = Real.cos θ) →
  θ = 7 * Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_from_point_l2800_280068


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2800_280078

theorem fractional_equation_solution : 
  ∃! x : ℝ, (x ≠ 0 ∧ x ≠ 2) ∧ (5 / x = 7 / (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2800_280078


namespace NUMINAMATH_CALUDE_max_scores_is_45_l2800_280070

/-- Represents a test with multiple-choice questions. -/
structure Test where
  num_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  unanswered_points : ℤ

/-- Calculates the maximum number of different possible total scores for a given test. -/
def max_different_scores (t : Test) : ℕ :=
  sorry

/-- The specific test described in the problem. -/
def problem_test : Test :=
  { num_questions := 10
  , correct_points := 4
  , incorrect_points := -1
  , unanswered_points := 0 }

/-- Theorem stating that the maximum number of different possible total scores for the problem_test is 45. -/
theorem max_scores_is_45 : max_different_scores problem_test = 45 := by
  sorry

end NUMINAMATH_CALUDE_max_scores_is_45_l2800_280070


namespace NUMINAMATH_CALUDE_larger_number_problem_l2800_280006

theorem larger_number_problem (x y : ℝ) : 
  y = 2 * x - 3 → x + y = 51 → max x y = 33 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2800_280006


namespace NUMINAMATH_CALUDE_contrapositive_square_sum_zero_l2800_280050

theorem contrapositive_square_sum_zero (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_square_sum_zero_l2800_280050


namespace NUMINAMATH_CALUDE_carson_counted_six_clouds_l2800_280059

/-- The number of clouds Carson counted that look like funny animals -/
def carson_clouds : ℕ := sorry

/-- The number of clouds Carson's little brother counted that look like dragons -/
def brother_clouds : ℕ := sorry

/-- The total number of clouds counted -/
def total_clouds : ℕ := 24

theorem carson_counted_six_clouds :
  carson_clouds = 6 ∧
  brother_clouds = 3 * carson_clouds ∧
  carson_clouds + brother_clouds = total_clouds :=
sorry

end NUMINAMATH_CALUDE_carson_counted_six_clouds_l2800_280059


namespace NUMINAMATH_CALUDE_parabola_sum_l2800_280082

/-- A parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_sum (p : Parabola) :
  p.x_coord (-6) = 7 →  -- vertex condition
  p.x_coord 0 = 5 →     -- point condition
  p.a + p.b + p.c = -32/3 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_l2800_280082


namespace NUMINAMATH_CALUDE_book_pages_l2800_280087

/-- The number of pages Frank reads per day -/
def pages_per_day : ℕ := 22

/-- The number of days it took Frank to finish the book -/
def days_to_finish : ℕ := 569

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_per_day * days_to_finish

theorem book_pages : total_pages = 12518 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_l2800_280087


namespace NUMINAMATH_CALUDE_dans_cards_l2800_280067

theorem dans_cards (initial_cards : ℕ) (bought_cards : ℕ) (total_cards : ℕ) : 
  initial_cards = 27 → bought_cards = 20 → total_cards = 88 → 
  total_cards - bought_cards - initial_cards = 41 := by
sorry

end NUMINAMATH_CALUDE_dans_cards_l2800_280067


namespace NUMINAMATH_CALUDE_new_person_weight_l2800_280079

/-- Given a group of 6 persons where replacing a 65 kg person with a new person
    increases the average weight by 1.5 kg, prove that the weight of the new person is 74 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_replaced : ℝ) (avg_increase : ℝ) :
  initial_count = 6 →
  weight_replaced = 65 →
  avg_increase = 1.5 →
  (initial_count : ℝ) * avg_increase + weight_replaced = 74 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2800_280079


namespace NUMINAMATH_CALUDE_range_of_a_l2800_280010

/-- Given a function f(x) = x^2 + 2(a-1)x + 2 that is monotonically decreasing on (-∞, 4],
    prove that the range of a is (-∞, -3]. -/
theorem range_of_a (a : ℝ) : 
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → (x^2 + 2*(a-1)*x + 2) > (y^2 + 2*(a-1)*y + 2)) →
  a ∈ Set.Iic (-3 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2800_280010


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l2800_280073

theorem smallest_number_with_remainders : ∃! n : ℕ,
  (∀ k ∈ Finset.range 10, n % (k + 3) = k + 2) ∧
  (∀ m : ℕ, m < n → ∃ k ∈ Finset.range 10, m % (k + 3) ≠ k + 2) :=
by
  use 27719
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l2800_280073


namespace NUMINAMATH_CALUDE_taxi_driver_probability_l2800_280081

-- Define the number of checkpoints
def num_checkpoints : ℕ := 6

-- Define the probability of encountering a red light at each checkpoint
def red_light_prob : ℚ := 1/3

-- Define the probability of passing exactly two checkpoints before encountering a red light
def pass_two_checkpoints_prob : ℚ := 4/27

-- State the theorem
theorem taxi_driver_probability :
  ∀ (n : ℕ) (p : ℚ),
  n = num_checkpoints →
  p = red_light_prob →
  pass_two_checkpoints_prob = (1 - p) * (1 - p) * p :=
by sorry

end NUMINAMATH_CALUDE_taxi_driver_probability_l2800_280081


namespace NUMINAMATH_CALUDE_compare_exponential_and_quadratic_l2800_280016

theorem compare_exponential_and_quadratic (n : ℕ) :
  (n ≥ 3 → 2^(2*n) > (2*n + 1)^2) ∧
  ((n = 1 ∨ n = 2) → 2^(2*n) < (2*n + 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_compare_exponential_and_quadratic_l2800_280016


namespace NUMINAMATH_CALUDE_other_x_intercept_l2800_280020

/-- A quadratic function with vertex (5, -3) and one x-intercept at (0, 0) -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem other_x_intercept (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x = a * (x - 5)^2 - 3) →  -- vertex form
  QuadraticFunction a b c 0 = 0 →                        -- (0, 0) is an x-intercept
  ∃ x, x ≠ 0 ∧ QuadraticFunction a b c x = 0 ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_other_x_intercept_l2800_280020


namespace NUMINAMATH_CALUDE_eve_gift_cost_l2800_280075

def hand_mitts_cost : ℝ := 14
def apron_cost : ℝ := 16
def utensils_cost : ℝ := 10
def knife_cost : ℝ := 2 * utensils_cost
def discount_percentage : ℝ := 0.25
def num_nieces : ℕ := 3

def total_cost_per_niece : ℝ := hand_mitts_cost + apron_cost + utensils_cost + knife_cost

def total_cost_before_discount : ℝ := num_nieces * total_cost_per_niece

def discount_amount : ℝ := discount_percentage * total_cost_before_discount

theorem eve_gift_cost : total_cost_before_discount - discount_amount = 135 := by
  sorry

end NUMINAMATH_CALUDE_eve_gift_cost_l2800_280075


namespace NUMINAMATH_CALUDE_f_continuous_l2800_280034

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + 3*x + 5

-- State the theorem
theorem f_continuous : Continuous f := by sorry

end NUMINAMATH_CALUDE_f_continuous_l2800_280034


namespace NUMINAMATH_CALUDE_cycle_selling_price_l2800_280056

/-- Given a cycle bought for a certain price with a specific gain percent,
    calculate the selling price. -/
def selling_price (cost_price : ℚ) (gain_percent : ℚ) : ℚ :=
  cost_price * (1 + gain_percent / 100)

/-- Theorem: The selling price of a cycle bought for Rs. 675 with a 60% gain is Rs. 1080 -/
theorem cycle_selling_price :
  selling_price 675 60 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l2800_280056


namespace NUMINAMATH_CALUDE_function_minimum_value_l2800_280000

theorem function_minimum_value (x : ℝ) (h : x > -1) :
  (x^2 + 7*x + 10) / (x + 1) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_value_l2800_280000


namespace NUMINAMATH_CALUDE_sunlight_rice_yield_correlation_l2800_280028

-- Define the concept of a relationship between two variables
def Relationship (X Y : Type) := X → Y → Prop

-- Define functional relationship
def FunctionalRelationship (X Y : Type) (r : Relationship X Y) : Prop :=
  ∀ (x : X), ∃! (y : Y), r x y

-- Define correlation relationship
def CorrelationRelationship (X Y : Type) (r : Relationship X Y) : Prop :=
  ¬FunctionalRelationship X Y r ∧ ∃ (x₁ x₂ : X) (y₁ y₂ : Y), r x₁ y₁ ∧ r x₂ y₂

-- Define the relationships for each option
def CubeVolumeEdgeLength : Relationship ℝ ℝ := sorry
def AngleSine : Relationship ℝ ℝ := sorry
def SunlightRiceYield : Relationship ℝ ℝ := sorry
def HeightVision : Relationship ℝ ℝ := sorry

-- State the theorem
theorem sunlight_rice_yield_correlation :
  CorrelationRelationship ℝ ℝ SunlightRiceYield ∧
  ¬CorrelationRelationship ℝ ℝ CubeVolumeEdgeLength ∧
  ¬CorrelationRelationship ℝ ℝ AngleSine ∧
  ¬CorrelationRelationship ℝ ℝ HeightVision :=
sorry

end NUMINAMATH_CALUDE_sunlight_rice_yield_correlation_l2800_280028


namespace NUMINAMATH_CALUDE_grid_erasing_game_strategies_l2800_280024

/-- Represents the possible outcomes of the grid erasing game -/
inductive GameOutcome
  | FirstPlayerWins
  | SecondPlayerWins

/-- Defines the grid erasing game -/
def GridErasingGame (rows : Nat) (cols : Nat) : GameOutcome :=
  sorry

/-- Theorem stating the winning strategies for different grid sizes -/
theorem grid_erasing_game_strategies :
  (GridErasingGame 10 12 = GameOutcome.SecondPlayerWins) ∧
  (GridErasingGame 9 10 = GameOutcome.FirstPlayerWins) ∧
  (GridErasingGame 9 11 = GameOutcome.SecondPlayerWins) := by
  sorry

/-- Lemma: In a grid with even dimensions, the second player has a winning strategy -/
lemma even_dimensions_second_player_wins (m n : Nat) 
  (hm : Even m) (hn : Even n) : 
  GridErasingGame m n = GameOutcome.SecondPlayerWins := by
  sorry

/-- Lemma: In a grid with one odd and one even dimension, the first player has a winning strategy -/
lemma odd_even_dimensions_first_player_wins (m n : Nat) 
  (hm : Odd m) (hn : Even n) : 
  GridErasingGame m n = GameOutcome.FirstPlayerWins := by
  sorry

/-- Lemma: In a grid with both odd dimensions, the second player has a winning strategy -/
lemma odd_dimensions_second_player_wins (m n : Nat) 
  (hm : Odd m) (hn : Odd n) : 
  GridErasingGame m n = GameOutcome.SecondPlayerWins := by
  sorry

end NUMINAMATH_CALUDE_grid_erasing_game_strategies_l2800_280024


namespace NUMINAMATH_CALUDE_income_ratio_l2800_280065

/-- Represents the financial data of a person --/
structure PersonFinance where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- The problem setup --/
def problem_setup (p1 p2 : PersonFinance) : Prop :=
  p1.income = 4000 ∧
  p1.savings = 1600 ∧
  p2.savings = 1600 ∧
  3 * p2.expenditure = 2 * p1.expenditure ∧
  p1.savings = p1.income - p1.expenditure ∧
  p2.savings = p2.income - p2.expenditure

/-- The theorem to be proved --/
theorem income_ratio (p1 p2 : PersonFinance) :
  problem_setup p1 p2 → 5 * p2.income = 4 * p1.income :=
by
  sorry


end NUMINAMATH_CALUDE_income_ratio_l2800_280065


namespace NUMINAMATH_CALUDE_xy_value_l2800_280030

theorem xy_value (x y : ℝ) (h_distinct : x ≠ y) 
  (h_eq : x^2 + 2/x^2 = y^2 + 2/y^2) : 
  x * y = Real.sqrt 2 ∨ x * y = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2800_280030
