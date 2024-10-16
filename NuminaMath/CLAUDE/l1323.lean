import Mathlib

namespace NUMINAMATH_CALUDE_angle_inequality_l1323_132308

theorem angle_inequality (θ : Real) : 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 2 → 
    x^2 * Real.sin θ - x * (2 - x) + (2 - x)^2 * Real.cos θ > 0) ↔ 
  (π / 12 < θ ∧ θ < 5 * π / 12) := by
  sorry

end NUMINAMATH_CALUDE_angle_inequality_l1323_132308


namespace NUMINAMATH_CALUDE_direction_vector_coefficient_l1323_132339

/-- Given a line passing through points (-2, 5) and (1, 0), prove that its direction vector of the form (a, -1) has a = 3/5 -/
theorem direction_vector_coefficient (p1 p2 : ℝ × ℝ) (a : ℝ) : 
  p1 = (-2, 5) → p2 = (1, 0) → 
  (p2.1 - p1.1, p2.2 - p1.2) = (3 * a, -3 * a) → 
  a = 3/5 := by sorry

end NUMINAMATH_CALUDE_direction_vector_coefficient_l1323_132339


namespace NUMINAMATH_CALUDE_polynomial_roots_l1323_132335

def polynomial (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2 - 4*x + 4

theorem polynomial_roots : 
  ∃ (a b c d e : ℝ), 
    (a = -1 - Real.sqrt 3) ∧
    (b = -1 + Real.sqrt 3) ∧
    (c = -1) ∧
    (d = 1) ∧
    (e = 2) ∧
    (∀ x : ℝ, polynomial x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l1323_132335


namespace NUMINAMATH_CALUDE_subset_condition_empty_intersection_l1323_132369

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

-- Theorem for part 1
theorem subset_condition (m : ℝ) : A ⊆ B m ↔ m ≤ -2 := by sorry

-- Theorem for part 2
theorem empty_intersection (m : ℝ) : A ∩ B m = ∅ ↔ 0 ≤ m := by sorry

end NUMINAMATH_CALUDE_subset_condition_empty_intersection_l1323_132369


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l1323_132313

/-- A parabola with equation y = -x^2 + 2x + m has its vertex on the x-axis if and only if m = -1 -/
theorem parabola_vertex_on_x_axis (m : ℝ) : 
  (∃ x, -x^2 + 2*x + m = 0 ∧ ∀ y, y = -x^2 + 2*x + m → y ≤ 0) ↔ m = -1 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l1323_132313


namespace NUMINAMATH_CALUDE_log_base_1024_integer_count_l1323_132329

theorem log_base_1024_integer_count :
  ∃! (S : Finset ℕ+), 
    (∀ b ∈ S, ∃ n : ℕ+, (b : ℝ) ^ (n : ℝ) = 1024) ∧ 
    (∀ b : ℕ+, (∃ n : ℕ+, (b : ℝ) ^ (n : ℝ) = 1024) → b ∈ S) ∧
    S.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_log_base_1024_integer_count_l1323_132329


namespace NUMINAMATH_CALUDE_equation_solutions_l1323_132305

theorem equation_solutions : 
  ∀ x y : ℤ, (3 : ℚ) / (x - 1) = (5 : ℚ) / (y - 2) ↔ (x = 1 ∧ y = 2) ∨ (x = 4 ∧ y = 7) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1323_132305


namespace NUMINAMATH_CALUDE_cross_section_area_l1323_132344

/-- Regular hexagonal pyramid with square lateral sides -/
structure HexagonalPyramid where
  /-- Side length of the base -/
  a : ℝ
  /-- Assumption that a is positive -/
  a_pos : 0 < a

/-- Cross-section of the hexagonal pyramid -/
def cross_section (pyramid : HexagonalPyramid) : Set (Fin 3 → ℝ) :=
  sorry

/-- Area of a set in ℝ² -/
noncomputable def area (s : Set (Fin 3 → ℝ)) : ℝ :=
  sorry

/-- Theorem: The area of the cross-section is 3a² -/
theorem cross_section_area (pyramid : HexagonalPyramid) :
    area (cross_section pyramid) = 3 * pyramid.a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_area_l1323_132344


namespace NUMINAMATH_CALUDE_total_seeds_planted_l1323_132382

/-- The number of tomato seeds planted by Mike and Ted -/
def total_seeds (mike_morning mike_afternoon ted_morning ted_afternoon : ℕ) : ℕ :=
  mike_morning + mike_afternoon + ted_morning + ted_afternoon

/-- Theorem stating the total number of tomato seeds planted by Mike and Ted -/
theorem total_seeds_planted : 
  ∀ (mike_morning mike_afternoon ted_morning ted_afternoon : ℕ),
  mike_morning = 50 →
  ted_morning = 2 * mike_morning →
  mike_afternoon = 60 →
  ted_afternoon = mike_afternoon - 20 →
  total_seeds mike_morning mike_afternoon ted_morning ted_afternoon = 250 :=
by
  sorry


end NUMINAMATH_CALUDE_total_seeds_planted_l1323_132382


namespace NUMINAMATH_CALUDE_puzzle_sets_theorem_l1323_132349

/-- Represents a set of puzzles --/
structure PuzzleSet where
  logic : ℕ
  visual : ℕ
  word : ℕ

/-- Checks if a puzzle set is valid according to the given conditions --/
def isValidSet (s : PuzzleSet) : Prop :=
  7 ≤ s.logic + s.visual + s.word
  ∧ s.logic + s.visual + s.word ≤ 12
  ∧ 3 * s.logic = 4 * s.visual
  ∧ 2 * s.word ≥ s.visual

/-- The maximum number of valid sets that can be created --/
def maxSets : ℕ := 5

/-- The theorem to be proved --/
theorem puzzle_sets_theorem :
  ∀ (s : PuzzleSet),
    isValidSet s →
    s.logic * maxSets ≤ 36 ∧
    s.visual * maxSets ≤ 27 ∧
    s.word * maxSets ≤ 15 ∧
    (∀ (n : ℕ), n > maxSets →
      ¬(s.logic * n ≤ 36 ∧
        s.visual * n ≤ 27 ∧
        s.word * n ≤ 15)) := by
  sorry

end NUMINAMATH_CALUDE_puzzle_sets_theorem_l1323_132349


namespace NUMINAMATH_CALUDE_price_change_theorem_l1323_132380

theorem price_change_theorem (initial_price : ℝ) 
  (jan_increase : ℝ) (feb_decrease : ℝ) (mar_increase : ℝ) (apr_decrease : ℝ) : 
  initial_price = 200 ∧ 
  jan_increase = 0.3 ∧ 
  feb_decrease = 0.1 ∧ 
  mar_increase = 0.2 ∧
  initial_price * (1 + jan_increase) * (1 - feb_decrease) * (1 + mar_increase) * (1 - apr_decrease) = initial_price →
  apr_decrease = 0.29 := by
  sorry

end NUMINAMATH_CALUDE_price_change_theorem_l1323_132380


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l1323_132343

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l1323_132343


namespace NUMINAMATH_CALUDE_system_solution_l1323_132348

theorem system_solution : ∃! (x y : ℝ), x + y = 2 ∧ 3 * x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1323_132348


namespace NUMINAMATH_CALUDE_problem_solution_l1323_132368

theorem problem_solution (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + 6 = 104 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1323_132368


namespace NUMINAMATH_CALUDE_decimal_difference_l1323_132336

-- Define the repeating decimal 0.2̅4̅
def repeating_decimal : ℚ := 8 / 33

-- Define the terminating decimal 0.24
def terminating_decimal : ℚ := 24 / 100

-- Theorem statement
theorem decimal_difference :
  repeating_decimal - terminating_decimal = 2 / 825 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l1323_132336


namespace NUMINAMATH_CALUDE_soccer_team_games_l1323_132312

theorem soccer_team_games (win lose tie rain higher : ℚ) 
  (ratio : win = 5.5 ∧ lose = 4.5 ∧ tie = 2.5 ∧ rain = 1 ∧ higher = 3.5)
  (lost_games : ℚ) (h_lost : lost_games = 13.5) :
  (win + lose + tie + rain + higher) * (lost_games / lose) = 51 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_games_l1323_132312


namespace NUMINAMATH_CALUDE_envelope_counting_time_l1323_132304

/-- Represents the time in seconds to count a given number of envelopes -/
def count_time (envelopes : ℕ) : ℕ :=
  if envelopes ≤ 100 then
    min ((100 - envelopes) / 10 * 10) (envelopes / 10 * 10)
  else
    envelopes / 10 * 10

theorem envelope_counting_time :
  (count_time 60 = 40) ∧ (count_time 90 = 10) := by
  sorry

end NUMINAMATH_CALUDE_envelope_counting_time_l1323_132304


namespace NUMINAMATH_CALUDE_min_value_on_circle_l1323_132385

theorem min_value_on_circle (x y : ℝ) : 
  x^2 + y^2 - 8*x + 6*y + 16 = 0 → ∃ (m : ℝ), m = 4 ∧ ∀ (a b : ℝ), a^2 + b^2 - 8*a + 6*b + 16 = 0 → x^2 + y^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l1323_132385


namespace NUMINAMATH_CALUDE_lego_count_l1323_132317

theorem lego_count (initial_legos : ℝ) (won_legos : ℝ) :
  initial_legos + won_legos = initial_legos + won_legos :=
by sorry

end NUMINAMATH_CALUDE_lego_count_l1323_132317


namespace NUMINAMATH_CALUDE_hyperbola_upper_focus_l1323_132371

/-- Given a hyperbola with equation y^2/16 - x^2/9 = 1, prove that the coordinates of the upper focus are (0, 5) -/
theorem hyperbola_upper_focus (x y : ℝ) :
  (y^2 / 16) - (x^2 / 9) = 1 →
  ∃ (a b c : ℝ),
    a = 4 ∧
    b = 3 ∧
    c^2 = a^2 + b^2 ∧
    (0, c) = (0, 5) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_upper_focus_l1323_132371


namespace NUMINAMATH_CALUDE_factor_implies_b_value_l1323_132356

/-- The polynomial Q(x) -/
def Q (b : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + b*x + 5

/-- Theorem: If x - 5 is a factor of Q(x), then b = -41 -/
theorem factor_implies_b_value (b : ℝ) : 
  (∀ x, Q b x = 0 ↔ x = 5) → b = -41 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_b_value_l1323_132356


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l1323_132353

/-- A game with vectors in a plane -/
structure VectorGame where
  n : ℕ
  vectors : Fin n → ℝ × ℝ

/-- The result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- A strategy for playing the game -/
def Strategy := (n : ℕ) → (remaining : Finset (Fin n)) → Fin n

/-- The game outcome given a strategy for the first player -/
def playGame (game : VectorGame) (strategy : Strategy) : GameResult :=
  sorry

/-- Theorem: The first player has a winning strategy -/
theorem first_player_winning_strategy (game : VectorGame) 
  (h : game.n = 2010) : 
  ∃ (strategy : Strategy), playGame game strategy = GameResult.FirstPlayerWins :=
sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l1323_132353


namespace NUMINAMATH_CALUDE_mike_height_l1323_132327

/-- Converts feet and inches to total inches -/
def to_inches (feet : ℕ) (inches : ℕ) : ℕ := feet * 12 + inches

/-- Converts total inches to feet and inches -/
def to_feet_and_inches (total_inches : ℕ) : ℕ × ℕ :=
  (total_inches / 12, total_inches % 12)

theorem mike_height (mark_feet : ℕ) (mark_inches : ℕ) (mike_inches : ℕ) :
  mark_feet = 5 →
  mark_inches = 3 →
  mike_inches = 1 →
  to_inches mark_feet mark_inches + 10 = to_inches 6 mike_inches :=
by sorry

end NUMINAMATH_CALUDE_mike_height_l1323_132327


namespace NUMINAMATH_CALUDE_greatest_common_divisor_under_30_l1323_132391

theorem greatest_common_divisor_under_30 : ∃ (n : ℕ), n = 18 ∧ 
  n ∣ 540 ∧ n < 30 ∧ n ∣ 180 ∧ 
  ∀ (m : ℕ), m ∣ 540 → m < 30 → m ∣ 180 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_under_30_l1323_132391


namespace NUMINAMATH_CALUDE_initial_items_count_l1323_132311

/-- The number of items Adam initially put in the shopping cart -/
def initial_items : ℕ := sorry

/-- The number of items Adam deleted from the shopping cart -/
def deleted_items : ℕ := 10

/-- The number of items left in Adam's shopping cart after deletion -/
def remaining_items : ℕ := 8

/-- Theorem stating that the initial number of items is 18 -/
theorem initial_items_count : initial_items = 18 :=
  by sorry

end NUMINAMATH_CALUDE_initial_items_count_l1323_132311


namespace NUMINAMATH_CALUDE_polynomial_expansion_problem_l1323_132370

theorem polynomial_expansion_problem (p q : ℝ) (hp : p > 0) (hq : q > 0) : 
  (45 * p^8 * q^2 = 120 * p^7 * q^3) → 
  (p + q = 3/4) → 
  p = 6/11 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_problem_l1323_132370


namespace NUMINAMATH_CALUDE_fraction_value_l1323_132394

theorem fraction_value (a b : ℝ) (h : a + 1/b = 2/a + 2*b ∧ a + 1/b ≠ 0) : a/b = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1323_132394


namespace NUMINAMATH_CALUDE_justin_run_time_l1323_132363

/-- Represents Justin's running speed and route information -/
structure RunningInfo where
  flat_speed : ℚ  -- blocks per minute on flat ground
  uphill_speed : ℚ  -- blocks per minute uphill
  total_distance : ℕ  -- total blocks to home
  uphill_distance : ℕ  -- blocks that are uphill

/-- Calculates the total time Justin needs to run home -/
def time_to_run_home (info : RunningInfo) : ℚ :=
  let flat_distance := info.total_distance - info.uphill_distance
  let flat_time := flat_distance / info.flat_speed
  let uphill_time := info.uphill_distance / info.uphill_speed
  flat_time + uphill_time

/-- Theorem stating that Justin will take 13 minutes to run home -/
theorem justin_run_time :
  let info : RunningInfo := {
    flat_speed := 1,  -- 2 blocks / 2 minutes
    uphill_speed := 2/3,  -- 2 blocks / 3 minutes
    total_distance := 10,
    uphill_distance := 6
  }
  time_to_run_home info = 13 := by
  sorry


end NUMINAMATH_CALUDE_justin_run_time_l1323_132363


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l1323_132350

-- Define the two fixed circles
def Q₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def Q₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

-- Define the property of being externally tangent
def externally_tangent (x y R : ℝ) : Prop :=
  ∀ (x₁ y₁ : ℝ), Q₁ x₁ y₁ → (x - x₁)^2 + (y - y₁)^2 = (1 + R)^2

-- Define the property of being internally tangent
def internally_tangent (x y R : ℝ) : Prop :=
  ∀ (x₂ y₂ : ℝ), Q₂ x₂ y₂ → (x - x₂)^2 + (y - y₂)^2 = (9 - R)^2

-- Define the trajectory of the center of the moving circle
def trajectory (x y : ℝ) : Prop := x^2/25 + y^2/16 = 1

-- State the theorem
theorem moving_circle_trajectory :
  ∀ (x y R : ℝ), externally_tangent x y R → internally_tangent x y R → trajectory x y :=
by sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l1323_132350


namespace NUMINAMATH_CALUDE_taehyung_candy_distribution_l1323_132328

/-- Given a total number of candies and the number of candies to be given to each friend,
    calculate the maximum number of friends who can receive candies. -/
def max_friends_with_candies (total_candies : ℕ) (candies_per_friend : ℕ) : ℕ :=
  total_candies / candies_per_friend

/-- Theorem: Given 45 candies and distributing 5 candies per friend,
    the maximum number of friends who can receive candies is 9. -/
theorem taehyung_candy_distribution :
  max_friends_with_candies 45 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_taehyung_candy_distribution_l1323_132328


namespace NUMINAMATH_CALUDE_photo_arrangements_l1323_132331

/-- The number of ways to arrange n distinct objects taken r at a time -/
def A (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of students -/
def total_students : ℕ := 7

/-- The number of students in the front row -/
def front_row : ℕ := 3

/-- The number of students in the back row -/
def back_row : ℕ := 4

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 3

/-- The number of spaces between boys -/
def spaces_between_boys : ℕ := 5

theorem photo_arrangements :
  (A total_students front_row * A back_row back_row = 5040) ∧
  (A front_row 1 * A back_row 1 * A (total_students - 2) (total_students - 2) = 1440) ∧
  (A (total_students - 2) (total_students - 2) * A 3 3 = 720) ∧
  (A num_boys num_boys * A spaces_between_boys num_girls = 1440) :=
sorry

end NUMINAMATH_CALUDE_photo_arrangements_l1323_132331


namespace NUMINAMATH_CALUDE_cone_to_cylinder_volume_ratio_l1323_132367

/-- The ratio of the volume of a cone to the volume of a cylinder with shared base radius -/
theorem cone_to_cylinder_volume_ratio 
  (r : ℝ) (h_c h_n : ℝ) 
  (hr : r = 5) 
  (hh_c : h_c = 20) 
  (hh_n : h_n = 10) : 
  (1 / 3 * π * r^2 * h_n) / (π * r^2 * h_c) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cone_to_cylinder_volume_ratio_l1323_132367


namespace NUMINAMATH_CALUDE_carnival_wait_time_l1323_132364

/-- Proves that the wait time for the roller coaster is 30 minutes given the carnival conditions --/
theorem carnival_wait_time (total_time : ℕ) (tilt_a_whirl_wait : ℕ) (giant_slide_wait : ℕ)
  (roller_coaster_rides : ℕ) (tilt_a_whirl_rides : ℕ) (giant_slide_rides : ℕ) :
  total_time = 4 * 60 ∧
  tilt_a_whirl_wait = 60 ∧
  giant_slide_wait = 15 ∧
  roller_coaster_rides = 4 ∧
  tilt_a_whirl_rides = 1 ∧
  giant_slide_rides = 4 →
  ∃ (roller_coaster_wait : ℕ),
    roller_coaster_wait = 30 ∧
    total_time = roller_coaster_rides * roller_coaster_wait +
                 tilt_a_whirl_rides * tilt_a_whirl_wait +
                 giant_slide_rides * giant_slide_wait :=
by
  sorry

end NUMINAMATH_CALUDE_carnival_wait_time_l1323_132364


namespace NUMINAMATH_CALUDE_stair_step_black_squares_l1323_132374

/-- Represents the number of squares added to form a row in the stair-step pattern -/
def squaresAdded (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 2

/-- Calculates the total number of squares in the nth row of the stair-step pattern -/
def totalSquares (n : ℕ) : ℕ :=
  1 + (Finset.range n).sum squaresAdded

/-- Calculates the number of black squares in a row with a given total number of squares -/
def blackSquares (total : ℕ) : ℕ :=
  (total - 1) / 2

/-- Theorem: The 20th row of the stair-step pattern contains 85 black squares -/
theorem stair_step_black_squares :
  blackSquares (totalSquares 20) = 85 := by
  sorry


end NUMINAMATH_CALUDE_stair_step_black_squares_l1323_132374


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1323_132376

theorem complex_modulus_problem : 
  Complex.abs ((1 + Complex.I) * (2 - Complex.I)) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1323_132376


namespace NUMINAMATH_CALUDE_equation_solutions_l1323_132354

theorem equation_solutions : ∀ x : ℝ, 3 * x^2 - 27 = 0 ↔ x = 3 ∨ x = -3 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1323_132354


namespace NUMINAMATH_CALUDE_sum_x_coordinates_of_Q3_l1323_132301

/-- A polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Create a new polygon from the midpoints of another polygon's sides -/
def midpointPolygon (p : Polygon) : Polygon :=
  sorry

/-- The sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

theorem sum_x_coordinates_of_Q3 (Q1 : Polygon) 
  (h1 : Q1.vertices.length = 45)
  (h2 : sumXCoordinates Q1 = 180) :
  let Q2 := midpointPolygon Q1
  let Q3 := midpointPolygon Q2
  sumXCoordinates Q3 = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_coordinates_of_Q3_l1323_132301


namespace NUMINAMATH_CALUDE_cooking_and_weaving_count_l1323_132309

/-- Represents the number of people in various curriculum combinations -/
structure CurriculumParticipation where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cooking_only : ℕ
  cooking_and_yoga : ℕ
  all_curriculums : ℕ

/-- Theorem stating the number of people studying both cooking and weaving -/
theorem cooking_and_weaving_count (cp : CurriculumParticipation)
  (h1 : cp.yoga = 25)
  (h2 : cp.cooking = 15)
  (h3 : cp.weaving = 8)
  (h4 : cp.cooking_only = 2)
  (h5 : cp.cooking_and_yoga = 7)
  (h6 : cp.all_curriculums = 3) :
  cp.cooking - cp.cooking_only - (cp.cooking_and_yoga - cp.all_curriculums) = 9 := by
  sorry


end NUMINAMATH_CALUDE_cooking_and_weaving_count_l1323_132309


namespace NUMINAMATH_CALUDE_ratio_problem_l1323_132398

theorem ratio_problem : ∀ x : ℚ, (20 : ℚ) / 1 = x / 10 → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1323_132398


namespace NUMINAMATH_CALUDE_cubic_sum_simplification_l1323_132323

theorem cubic_sum_simplification (a b : ℝ) : 
  a^2 = 9/25 → 
  b^2 = (3 + Real.sqrt 3)^2 / 15 → 
  a < 0 → 
  b > 0 → 
  (a + b)^3 = (-5670 * Real.sqrt 3 + 1620 * Real.sqrt 5 + 15 * Real.sqrt 15) / 50625 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_simplification_l1323_132323


namespace NUMINAMATH_CALUDE_pyarelal_loss_l1323_132388

theorem pyarelal_loss (p a : ℝ) (total_loss : ℝ) : 
  a = (1 / 9) * p → 
  total_loss = 900 → 
  (p / (p + a)) * total_loss = 810 :=
by sorry

end NUMINAMATH_CALUDE_pyarelal_loss_l1323_132388


namespace NUMINAMATH_CALUDE_maggie_bouncy_balls_l1323_132300

/-- The number of packs of green bouncy balls Maggie gave away -/
def green_packs_given_away : ℝ := 4.0

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs_bought : ℝ := 8.0

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs_bought : ℝ := 4.0

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℝ := 10.0

/-- The total number of bouncy balls Maggie kept -/
def balls_kept : ℕ := 80

theorem maggie_bouncy_balls :
  green_packs_given_away = 
    ((yellow_packs_bought + green_packs_bought) * balls_per_pack - balls_kept) / balls_per_pack :=
by sorry

end NUMINAMATH_CALUDE_maggie_bouncy_balls_l1323_132300


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l1323_132345

def f (x : ℝ) := abs x + 1

theorem f_even_and_increasing :
  (∀ x, f x = f (-x)) ∧
  (∀ x y, 0 < x → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l1323_132345


namespace NUMINAMATH_CALUDE_no_real_zeros_l1323_132389

theorem no_real_zeros (x : ℝ) : x^6 - x^5 + x^4 - x^3 + x^2 - x + 3/4 ≥ 3/8 := by
  sorry

end NUMINAMATH_CALUDE_no_real_zeros_l1323_132389


namespace NUMINAMATH_CALUDE_jason_initial_quarters_l1323_132381

/-- The number of quarters Jason's dad gave him -/
def quarters_from_dad : ℕ := 25

/-- The total number of quarters Jason has now -/
def total_quarters_now : ℕ := 74

/-- The number of quarters Jason had initially -/
def initial_quarters : ℕ := total_quarters_now - quarters_from_dad

theorem jason_initial_quarters : initial_quarters = 49 := by
  sorry

end NUMINAMATH_CALUDE_jason_initial_quarters_l1323_132381


namespace NUMINAMATH_CALUDE_square_area_l1323_132341

theorem square_area (side : ℝ) (h : side = 6) : side * side = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l1323_132341


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l1323_132340

theorem cube_root_unity_sum (ω : ℂ) : 
  ω^3 = 1 → ω ≠ 1 → (2 - ω + ω^2)^4 + (2 + ω - ω^2)^4 = 512 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l1323_132340


namespace NUMINAMATH_CALUDE_max_min_product_l1323_132337

theorem max_min_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (sum_eq : x + y + z = 20) (prod_sum_eq : x*y + y*z + z*x = 78) :
  ∃ (M : ℝ), M = min (x*y) (min (y*z) (z*x)) ∧ M ≤ 400/9 ∧
  ∀ (M' : ℝ), (∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    x' + y' + z' = 20 ∧ x'*y' + y'*z' + z'*x' = 78 ∧
    M' = min (x'*y') (min (y'*z') (z'*x'))) → M' ≤ 400/9 :=
by sorry

end NUMINAMATH_CALUDE_max_min_product_l1323_132337


namespace NUMINAMATH_CALUDE_smallest_nonfactor_product_of_48_l1323_132351

theorem smallest_nonfactor_product_of_48 (a b : ℕ) : 
  a ≠ b ∧ 
  a > 0 ∧ 
  b > 0 ∧ 
  48 % a = 0 ∧ 
  48 % b = 0 ∧ 
  48 % (a * b) ≠ 0 →
  ∀ x y : ℕ, 
    x ≠ y ∧ 
    x > 0 ∧ 
    y > 0 ∧ 
    48 % x = 0 ∧ 
    48 % y = 0 ∧ 
    48 % (x * y) ≠ 0 →
    a * b ≤ x * y ∧
    a * b = 18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonfactor_product_of_48_l1323_132351


namespace NUMINAMATH_CALUDE_toms_average_strokes_l1323_132303

/-- Represents the number of rounds Tom plays -/
def rounds : ℕ := 9

/-- Represents the par value per hole -/
def par_per_hole : ℕ := 3

/-- Represents the number of strokes Tom was over par -/
def strokes_over_par : ℕ := 9

/-- Calculates Tom's average number of strokes per hole -/
def average_strokes_per_hole : ℚ :=
  (rounds * par_per_hole + strokes_over_par) / rounds

/-- Theorem stating that Tom's average number of strokes per hole is 4 -/
theorem toms_average_strokes :
  average_strokes_per_hole = 4 := by sorry

end NUMINAMATH_CALUDE_toms_average_strokes_l1323_132303


namespace NUMINAMATH_CALUDE_gecko_ratio_l1323_132378

-- Define the number of geckos sold last year
def geckos_last_year : ℕ := 86

-- Define the total number of geckos sold in the last two years
def total_geckos : ℕ := 258

-- Define the number of geckos sold the year before
def geckos_year_before : ℕ := total_geckos - geckos_last_year

-- Theorem to prove the ratio
theorem gecko_ratio : 
  geckos_year_before = 2 * geckos_last_year := by
  sorry

#check gecko_ratio

end NUMINAMATH_CALUDE_gecko_ratio_l1323_132378


namespace NUMINAMATH_CALUDE_bonnie_cupcakes_l1323_132358

/-- Represents the problem of calculating cupcakes to give to Bonnie -/
def cupcakes_to_give (total_goal : ℕ) (days : ℕ) (daily_goal : ℕ) : ℕ :=
  days * daily_goal - total_goal

theorem bonnie_cupcakes :
  cupcakes_to_give 96 2 60 = 24 := by
  sorry

end NUMINAMATH_CALUDE_bonnie_cupcakes_l1323_132358


namespace NUMINAMATH_CALUDE_exponent_division_rule_l1323_132330

theorem exponent_division_rule (a b : ℝ) (m : ℤ) 
  (ha : a > 0) (hb : b ≠ 0) : 
  (b / a) ^ m = a ^ (-m) * b ^ m := by sorry

end NUMINAMATH_CALUDE_exponent_division_rule_l1323_132330


namespace NUMINAMATH_CALUDE_solution_set_inequality_not_sufficient_condition_negation_of_proposition_not_necessary_condition_l1323_132322

-- Statement 1
theorem solution_set_inequality (x : ℝ) : 
  (x + 2) / (2 * x + 1) > 1 ↔ -1/2 < x ∧ x < 1 := by sorry

-- Statement 2
theorem not_sufficient_condition : 
  ∃ a b : ℝ, a * b > 1 ∧ ¬(a > 1 ∧ b > 1) := by sorry

-- Statement 3
theorem negation_of_proposition : 
  ¬(∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 := by sorry

-- Statement 4
theorem not_necessary_condition : 
  ∃ a : ℝ, a < 6 ∧ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_not_sufficient_condition_negation_of_proposition_not_necessary_condition_l1323_132322


namespace NUMINAMATH_CALUDE_max_quarters_and_dimes_l1323_132315

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The total amount Eva has in cents -/
def total_amount : ℕ := 480

/-- 
Given $4.80 in U.S. coins with an equal number of quarters and dimes,
prove that the maximum number of quarters (and dimes) is 13.
-/
theorem max_quarters_and_dimes :
  ∃ (n : ℕ), n * (quarter_value + dime_value) ≤ total_amount ∧
             ∀ (m : ℕ), m * (quarter_value + dime_value) ≤ total_amount → m ≤ n ∧
             n = 13 :=
sorry

end NUMINAMATH_CALUDE_max_quarters_and_dimes_l1323_132315


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1323_132392

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x > 1 → x^2 + x - 2 > 0) ∧
  (∃ x, x^2 + x - 2 > 0 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1323_132392


namespace NUMINAMATH_CALUDE_average_production_l1323_132326

/-- Given a production of 4000 items/month for 3 months and 4500 items/month for 9 months,
    the average production for 12 months is 4375 items/month. -/
theorem average_production (first_3_months : ℕ) (next_9_months : ℕ) (total_months : ℕ) :
  first_3_months = 3 →
  next_9_months = 9 →
  total_months = first_3_months + next_9_months →
  (first_3_months * 4000 + next_9_months * 4500) / total_months = 4375 :=
by sorry

end NUMINAMATH_CALUDE_average_production_l1323_132326


namespace NUMINAMATH_CALUDE_coin_toss_sequences_count_l1323_132321

/-- The number of ways to insert k items into n bins -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (n - 1)

/-- The number of different sequences of 17 coin tosses with specific subsequence counts -/
def coinTossSequences : ℕ := 
  let hh_insertions := starsAndBars 5 3  -- Insert 3 H into 5 existing H positions
  let tt_insertions := starsAndBars 4 6  -- Insert 6 T into 4 existing T positions
  hh_insertions * tt_insertions

/-- Theorem stating the number of coin toss sequences -/
theorem coin_toss_sequences_count :
  coinTossSequences = 2940 := by sorry

end NUMINAMATH_CALUDE_coin_toss_sequences_count_l1323_132321


namespace NUMINAMATH_CALUDE_exists_universal_program_l1323_132332

/- Define the maze structure -/
def Maze := Fin 10 → Fin 10 → Bool

/- Define the robot's position -/
structure Position where
  x : Fin 10
  y : Fin 10

/- Define the possible robot commands -/
inductive Command
| L
| R
| U
| D

/- Define a program as a list of commands -/
def Program := List Command

/- Function to check if a cell is accessible -/
def isAccessible (maze : Maze) (pos : Position) : Bool :=
  maze pos.x pos.y

/- Function to apply a command to a position -/
def applyCommand (maze : Maze) (pos : Position) (cmd : Command) : Position :=
  sorry

/- Function to check if a program visits all accessible cells -/
def visitsAllCells (maze : Maze) (start : Position) (prog : Program) : Prop :=
  sorry

/- The main theorem -/
theorem exists_universal_program :
  ∃ (prog : Program),
    ∀ (maze : Maze) (start : Position),
      visitsAllCells maze start prog :=
sorry

end NUMINAMATH_CALUDE_exists_universal_program_l1323_132332


namespace NUMINAMATH_CALUDE_cosine_sine_eighth_power_bounds_l1323_132383

theorem cosine_sine_eighth_power_bounds (x : ℝ) : 
  1/8 ≤ (Real.cos x)^8 + (Real.sin x)^8 ∧ (Real.cos x)^8 + (Real.sin x)^8 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_eighth_power_bounds_l1323_132383


namespace NUMINAMATH_CALUDE_aquafaba_needed_is_32_l1323_132316

/-- The number of tablespoons of aquafaba equivalent to one egg white -/
def aquafaba_per_egg : ℕ := 2

/-- The number of egg whites required for one angel food cake -/
def egg_whites_per_cake : ℕ := 8

/-- The number of angel food cakes Christine is making -/
def number_of_cakes : ℕ := 2

/-- The total number of tablespoons of aquafaba needed for the cakes -/
def total_aquafaba_needed : ℕ := aquafaba_per_egg * egg_whites_per_cake * number_of_cakes

theorem aquafaba_needed_is_32 : total_aquafaba_needed = 32 := by
  sorry

end NUMINAMATH_CALUDE_aquafaba_needed_is_32_l1323_132316


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l1323_132366

theorem triangle_angle_calculation (a b c A B C : Real) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  -- Given conditions
  (a = Real.sqrt 2) →
  (b = Real.sqrt 3) →
  (B = π / 3) →
  -- Law of Sines
  (a / Real.sin A = b / Real.sin B) →
  -- Conclusion
  A = π / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l1323_132366


namespace NUMINAMATH_CALUDE_income_tax_problem_l1323_132333

theorem income_tax_problem (q : ℝ) :
  let tax_rate_low := q / 100
  let tax_rate_high := (q + 3) / 100
  let total_tax_rate := (q + 0.5) / 100
  let income := 36000
  let tax_low := tax_rate_low * 30000
  let tax_high := tax_rate_high * (income - 30000)
  tax_low + tax_high = total_tax_rate * income := by sorry

end NUMINAMATH_CALUDE_income_tax_problem_l1323_132333


namespace NUMINAMATH_CALUDE_intersection_sum_l1323_132360

/-- Given two lines that intersect at (3,6), prove that a + b = 6 -/
theorem intersection_sum (a b : ℝ) : 
  (3 = (1/3) * 6 + a) → 
  (6 = (1/3) * 3 + b) → 
  a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l1323_132360


namespace NUMINAMATH_CALUDE_inequality_proof_l1323_132365

theorem inequality_proof (a b c d p q : ℝ) 
  (h1 : a * b + c * d = 2 * p * q) 
  (h2 : a * c ≥ p^2) 
  (h3 : p > 0) : 
  b * d ≤ q^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1323_132365


namespace NUMINAMATH_CALUDE_diana_earnings_l1323_132359

theorem diana_earnings (x : ℝ) 
  (july : x > 0)
  (august : x > 0)
  (september : x > 0)
  (total : x + 3*x + 6*x = 1500) : x = 150 := by
  sorry

end NUMINAMATH_CALUDE_diana_earnings_l1323_132359


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l1323_132393

/-- An isosceles trapezoid with the given measurements --/
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  longerBase : ℝ

/-- The area of an isosceles trapezoid --/
def trapezoidArea (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- The theorem stating the area of the specific trapezoid --/
theorem specific_trapezoid_area : 
  let t : IsoscelesTrapezoid := { 
    leg := 20,
    diagonal := 25,
    longerBase := 30
  }
  abs (trapezoidArea t - 315.82) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l1323_132393


namespace NUMINAMATH_CALUDE_shuttlecock_mass_probability_l1323_132302

variable (ξ : ℝ)

-- Define the probabilities given in the problem
def P_less_than_4_8 : ℝ := 0.3
def P_not_less_than_4_85 : ℝ := 0.32

-- Define the probability we want to prove
def P_between_4_8_and_4_85 : ℝ := 1 - P_less_than_4_8 - P_not_less_than_4_85

-- Theorem statement
theorem shuttlecock_mass_probability :
  P_between_4_8_and_4_85 = 0.38 := by sorry

end NUMINAMATH_CALUDE_shuttlecock_mass_probability_l1323_132302


namespace NUMINAMATH_CALUDE_f_satisfies_equation_l1323_132397

/-- The function f satisfying the given functional equation -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ -0.5 then 1 / (x + 0.5) else 0.5

/-- Theorem stating that f satisfies the functional equation for all real x -/
theorem f_satisfies_equation : ∀ x : ℝ, f x - (x - 0.5) * f (-x - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_equation_l1323_132397


namespace NUMINAMATH_CALUDE_vector_subtraction_magnitude_l1323_132357

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

theorem vector_subtraction_magnitude : ‖a - b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_magnitude_l1323_132357


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l1323_132347

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (k : ℝ), 5 * x^2 + 17 * x - 12 = (x + 4) * k := by
  sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l1323_132347


namespace NUMINAMATH_CALUDE_j_percentage_less_than_p_l1323_132386

/-- Given t = 6.25, t is t% less than p, and j is 20% less than t, prove j is 25% less than p -/
theorem j_percentage_less_than_p (t p j : ℝ) : 
  t = 6.25 →
  t = p * (100 - t) / 100 →
  j = t * 0.8 →
  j = p * 0.75 := by
  sorry

end NUMINAMATH_CALUDE_j_percentage_less_than_p_l1323_132386


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1323_132387

/-- Lateral surface area of a cone with given base radius and volume -/
theorem cone_lateral_surface_area (r h : ℝ) (hr : r = 3) (hv : (1/3) * π * r^2 * h = 12 * π) :
  π * r * (Real.sqrt (r^2 + h^2)) = 15 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1323_132387


namespace NUMINAMATH_CALUDE_max_a_inequality_max_a_is_five_l1323_132372

theorem max_a_inequality (a : ℝ) : 
  (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) → a ≤ 5 :=
by sorry

theorem max_a_is_five : 
  ∃ a : ℝ, (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) ∧ a = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_a_inequality_max_a_is_five_l1323_132372


namespace NUMINAMATH_CALUDE_part_one_part_two_l1323_132396

/-- Definition of the sequence sum -/
def S (n : ℕ) (a : ℝ) : ℝ := a * 2^n - 1

/-- Definition of the sequence terms -/
def a (n : ℕ) (a : ℝ) : ℝ := S n a - S (n-1) a

/-- Part 1: Prove the values of a_1 and a_4 when a = 3 -/
theorem part_one :
  a 1 3 = 5 ∧ a 4 3 = 24 :=
sorry

/-- Definition of geometric sequence -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n+1) = r * a n

/-- Part 2: Prove the value of a when {a_n} is a geometric sequence -/
theorem part_two :
  ∃ f : ℕ → ℝ, is_geometric_sequence f ∧ (∀ n : ℕ, S n 1 = f n - f 0) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1323_132396


namespace NUMINAMATH_CALUDE_bac_hex_to_decimal_l1323_132325

/-- Converts a hexadecimal digit to its decimal value -/
def hex_to_decimal (digit : Char) : ℕ :=
  match digit with
  | 'B' => 11
  | 'A' => 10
  | 'C' => 12
  | _ => 0  -- Default case, should not be reached for our specific problem

/-- Converts a three-digit hexadecimal number to decimal -/
def hex_to_decimal_3digit (h1 h2 h3 : Char) : ℕ :=
  (hex_to_decimal h1) * 256 + (hex_to_decimal h2) * 16 + (hex_to_decimal h3)

theorem bac_hex_to_decimal :
  hex_to_decimal_3digit 'B' 'A' 'C' = 2988 := by
  sorry

end NUMINAMATH_CALUDE_bac_hex_to_decimal_l1323_132325


namespace NUMINAMATH_CALUDE_stock_price_calculation_l1323_132319

def initial_price : ℝ := 120
def first_year_increase : ℝ := 0.8
def second_year_decrease : ℝ := 0.3

theorem stock_price_calculation :
  let price_after_first_year := initial_price * (1 + first_year_increase)
  let final_price := price_after_first_year * (1 - second_year_decrease)
  final_price = 151.2 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l1323_132319


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1323_132318

theorem fraction_to_decimal : (45 : ℚ) / (2^3 * 5^4) = 0.0090 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1323_132318


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l1323_132338

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_odd_sum : a 1 + a 3 + a 5 + a 7 + a 9 = 10)
  (h_even_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 20) :
  a 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l1323_132338


namespace NUMINAMATH_CALUDE_arrangement_count_l1323_132334

theorem arrangement_count (n m : ℕ) (hn : n = 6) (hm : m = 4) :
  (Nat.choose n m) * (Nat.factorial m) = 360 :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_l1323_132334


namespace NUMINAMATH_CALUDE_circular_track_time_theorem_l1323_132373

/-- Represents a circular track with two points -/
structure CircularTrack :=
  (total_time : ℝ)
  (time_closer_to_point : ℝ)

/-- Theorem: If a runner on a circular track is closer to one point for half the total running time,
    then the total running time is twice the time the runner is closer to that point -/
theorem circular_track_time_theorem (track : CircularTrack) 
  (h1 : track.time_closer_to_point > 0)
  (h2 : track.time_closer_to_point = track.total_time / 2) : 
  track.total_time = 2 * track.time_closer_to_point :=
sorry

end NUMINAMATH_CALUDE_circular_track_time_theorem_l1323_132373


namespace NUMINAMATH_CALUDE_quadratic_composition_theorem_l1323_132399

/-- A unitary quadratic trinomial -/
structure UnitaryQuadratic where
  b : ℝ
  c : ℝ

/-- Evaluate a unitary quadratic trinomial at a point -/
def evaluate (f : UnitaryQuadratic) (x : ℝ) : ℝ :=
  x^2 + f.b * x + f.c

/-- Composition of two unitary quadratic trinomials -/
def compose (f g : UnitaryQuadratic) : UnitaryQuadratic :=
  { b := g.b^2 + f.b * (1 + g.b) + g.c * f.b
    c := g.c^2 + f.b * g.c + f.c }

/-- A polynomial has no real roots -/
def hasNoRealRoots (f : UnitaryQuadratic) : Prop :=
  ∀ x : ℝ, evaluate f x ≠ 0

theorem quadratic_composition_theorem (f g : UnitaryQuadratic) 
    (h1 : hasNoRealRoots (compose f g))
    (h2 : hasNoRealRoots (compose g f)) :
    hasNoRealRoots (compose f f) ∨ hasNoRealRoots (compose g g) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_composition_theorem_l1323_132399


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1323_132320

-- Define the concept of a function being even or odd
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the condition that both f and g are either odd or even
def BothEvenOrOdd (f g : ℝ → ℝ) : Prop :=
  (IsEven f ∧ IsEven g) ∨ (IsOdd f ∧ IsOdd g)

-- Define the property that the product of f and g is even
def ProductIsEven (f g : ℝ → ℝ) : Prop :=
  IsEven (fun x ↦ f x * g x)

-- Theorem statement
theorem sufficient_not_necessary (f g : ℝ → ℝ) :
  (BothEvenOrOdd f g → ProductIsEven f g) ∧
  ¬(ProductIsEven f g → BothEvenOrOdd f g) := by
  sorry


end NUMINAMATH_CALUDE_sufficient_not_necessary_l1323_132320


namespace NUMINAMATH_CALUDE_min_total_routes_l1323_132342

/-- Represents the number of routes for each airline company -/
structure AirlineRoutes where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The minimum number of routes needed to maintain connectivity -/
def min_connectivity : ℕ := 14

/-- The total number of cities in the country -/
def num_cities : ℕ := 15

/-- Predicate to check if the network remains connected after removing any one company's routes -/
def remains_connected (routes : AirlineRoutes) : Prop :=
  routes.a + routes.b ≥ min_connectivity ∧
  routes.b + routes.c ≥ min_connectivity ∧
  routes.c + routes.a ≥ min_connectivity

/-- Theorem stating the minimum number of total routes needed -/
theorem min_total_routes (routes : AirlineRoutes) :
  remains_connected routes → routes.a + routes.b + routes.c ≥ 21 := by
  sorry


end NUMINAMATH_CALUDE_min_total_routes_l1323_132342


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l1323_132379

/-- Given a triangle with two known angles of 45° and 70°, prove that the third angle is 65° and the largest angle is 70°. -/
theorem triangle_angle_proof (a b c : ℝ) : 
  a = 45 → b = 70 → a + b + c = 180 → 
  c = 65 ∧ max a (max b c) = 70 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l1323_132379


namespace NUMINAMATH_CALUDE_blue_pill_cost_is_21_l1323_132361

/-- The cost of a blue pill given the conditions of Ben's medication regimen -/
def blue_pill_cost (total_cost : ℚ) (duration_days : ℕ) (blue_red_diff : ℚ) : ℚ :=
  let daily_cost : ℚ := total_cost / duration_days
  let x : ℚ := (daily_cost + blue_red_diff) / 2
  x

theorem blue_pill_cost_is_21 :
  blue_pill_cost 819 21 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_blue_pill_cost_is_21_l1323_132361


namespace NUMINAMATH_CALUDE_sum_of_segments_eq_165_l1323_132395

/-- The sum of lengths of all possible line segments formed by dividing a line segment of length 9 into 9 equal parts -/
def sum_of_segments : ℕ :=
  let n : ℕ := 9  -- number of divisions
  (n * (n + 1) * (n + 2)) / 6

/-- Theorem: The sum of lengths of all possible line segments formed by dividing a line segment of length 9 into 9 equal parts is equal to 165 -/
theorem sum_of_segments_eq_165 : sum_of_segments = 165 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_segments_eq_165_l1323_132395


namespace NUMINAMATH_CALUDE_expression_evaluation_l1323_132324

theorem expression_evaluation (a b : ℝ) (h : (a + 1)^2 + |b + 1| = 0) :
  1 - (a^2 + 2*a*b + b^2) / (a^2 - a*b) / ((a + b) / (a - b)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1323_132324


namespace NUMINAMATH_CALUDE_log_sum_simplification_l1323_132306

theorem log_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_simplification_l1323_132306


namespace NUMINAMATH_CALUDE_fraction_equality_l1323_132307

theorem fraction_equality : (5 * 7 - 3) / 9 = 32 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1323_132307


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l1323_132310

/-- Given a triangle ABC with sides a, b, c in the ratio 2:3:4, 
    it is not necessarily a right triangle -/
theorem not_necessarily_right_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (ratio : ∃ (k : ℝ), k > 0 ∧ a = 2*k ∧ b = 3*k ∧ c = 4*k) : 
  ¬(a^2 + b^2 = c^2) := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l1323_132310


namespace NUMINAMATH_CALUDE_percentage_problem_l1323_132314

theorem percentage_problem (P : ℝ) : 
  (0.15 * P * (0.5 * 5600) = 126) → P = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1323_132314


namespace NUMINAMATH_CALUDE_shorter_tree_height_l1323_132377

theorem shorter_tree_height (h1 h2 : ℝ) : 
  h2 = h1 + 20 →  -- One tree is 20 feet taller
  h1 / h2 = 5 / 7 →  -- Heights are in ratio 5:7
  h1 + h2 = 240 →  -- Sum of heights is 240 feet
  h1 = 110 :=  -- Shorter tree is 110 feet tall
by sorry

end NUMINAMATH_CALUDE_shorter_tree_height_l1323_132377


namespace NUMINAMATH_CALUDE_gcd_180_450_l1323_132352

theorem gcd_180_450 : Nat.gcd 180 450 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_450_l1323_132352


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1323_132390

theorem complex_equation_solution (i : ℂ) (m : ℝ) :
  i * i = -1 →
  (1 - m * i) / (i^3) = 1 + i →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1323_132390


namespace NUMINAMATH_CALUDE_complex_imaginary_condition_l1323_132362

theorem complex_imaginary_condition (a : ℝ) : 
  (Complex.I.re = 0 ∧ Complex.I.im = 1) →
  ((1 - 2 * Complex.I) * (a + Complex.I)).re = 0 →
  a = -2 :=
sorry

end NUMINAMATH_CALUDE_complex_imaginary_condition_l1323_132362


namespace NUMINAMATH_CALUDE_smallest_k_for_sum_and_product_existence_of_solution_smallest_k_is_four_l1323_132384

theorem smallest_k_for_sum_and_product (k : ℝ) : 
  (k > 0 ∧ 
   ∃ a b : ℝ, a + b = k ∧ a * b = k) → 
  k ≥ 4 :=
by sorry

theorem existence_of_solution : 
  ∃ k a b : ℝ, k > 0 ∧ a + b = k ∧ a * b = k ∧ k = 4 :=
by sorry

theorem smallest_k_is_four : 
  ∃! k : ℝ, k > 0 ∧ 
  (∃ a b : ℝ, a + b = k ∧ a * b = k) ∧
  (∀ k' : ℝ, k' > 0 → (∃ a b : ℝ, a + b = k' ∧ a * b = k') → k' ≥ k) ∧
  k = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_sum_and_product_existence_of_solution_smallest_k_is_four_l1323_132384


namespace NUMINAMATH_CALUDE_tangent_line_and_special_points_l1323_132355

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 16
def C₂ (x y : ℝ) : Prop := (x-7)^2 + (y-4)^2 = 4

-- Define the tangent line condition
def is_tangent_line (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, C₁ x y → (a*x + b*y + c = 0 → 
    ∀ x' y' : ℝ, C₁ x' y' → a*x' + b*y' + c ≥ 0)

-- Define the perpendicular lines condition
def perpendicular_lines_condition (a b : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ 
    (∀ x y : ℝ, (y - b = k*(x - a) → C₁ x y) ↔ 
                (x + k*y - b*k - a = 0 → C₂ x y)) ∧
    (∀ x₁ y₁ x₂ y₂ : ℝ, 
      C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧ y₁ - b = k*(x₁ - a) ∧ y₂ - b = k*(x₂ - a) →
      ∃ x₃ y₃ x₄ y₄ : ℝ, 
        C₂ x₃ y₃ ∧ C₂ x₄ y₄ ∧ 
        x₃ + k*y₃ - b*k - a = 0 ∧ x₄ + k*y₄ - b*k - a = 0 ∧
        (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4*((x₃ - x₄)^2 + (y₃ - y₄)^2))

theorem tangent_line_and_special_points :
  (is_tangent_line 5 (-12) 52 ∨ is_tangent_line 1 0 (-4)) ∧
  (perpendicular_lines_condition 4 6 ∧ perpendicular_lines_condition (36/5) (2/5)) ∧
  (∀ a b : ℝ, perpendicular_lines_condition a b → 
    (a = 4 ∧ b = 6) ∨ (a = 36/5 ∧ b = 2/5)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_special_points_l1323_132355


namespace NUMINAMATH_CALUDE_yoongi_initial_books_l1323_132346

/-- Represents the number of books each person has -/
structure BookCount where
  yoongi : ℕ
  eunji : ℕ
  yuna : ℕ

/-- Represents the book exchange described in the problem -/
def exchange (initial : BookCount) : BookCount :=
  { yoongi := initial.yoongi - 5 + 15,
    eunji := initial.eunji + 5 - 10,
    yuna := initial.yuna + 10 - 15 }

/-- Theorem stating that if after the exchange all have 45 books, 
    Yoongi must have started with 35 books -/
theorem yoongi_initial_books 
  (initial : BookCount) 
  (h : exchange initial = {yoongi := 45, eunji := 45, yuna := 45}) : 
  initial.yoongi = 35 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_initial_books_l1323_132346


namespace NUMINAMATH_CALUDE_sixth_grade_boys_count_l1323_132375

/-- Represents the set of boys in the 6th "A" grade. -/
def Boys : Type := Unit

/-- Represents the set of girls in the 6th "A" grade. -/
inductive Girls : Type
  | tanya : Girls
  | dasha : Girls
  | katya : Girls

/-- Represents the friendship relation between boys and girls. -/
def IsFriend : Boys → Girls → Prop := sorry

/-- The number of boys in the 6th "A" grade. -/
def numBoys : ℕ := sorry

theorem sixth_grade_boys_count :
  (∀ (b1 b2 b3 : Boys), ∃ (g : Girls), IsFriend b1 g ∨ IsFriend b2 g ∨ IsFriend b3 g) →
  (∃ (boys : Finset Boys), Finset.card boys = 12 ∧ ∀ b ∈ boys, IsFriend b Girls.tanya) →
  (∃ (boys : Finset Boys), Finset.card boys = 12 ∧ ∀ b ∈ boys, IsFriend b Girls.dasha) →
  (∃ (boys : Finset Boys), Finset.card boys = 13 ∧ ∀ b ∈ boys, IsFriend b Girls.katya) →
  numBoys = 13 ∨ numBoys = 14 := by
  sorry

end NUMINAMATH_CALUDE_sixth_grade_boys_count_l1323_132375
