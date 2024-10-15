import Mathlib

namespace NUMINAMATH_CALUDE_number_equation_l1263_126349

theorem number_equation (x : ℝ) : 3 * x = (26 - x) + 10 ↔ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l1263_126349


namespace NUMINAMATH_CALUDE_train_speed_l1263_126390

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 250) (h2 : time = 12) :
  ∃ (speed : ℝ), abs (speed - length / time) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1263_126390


namespace NUMINAMATH_CALUDE_lcm_problem_l1263_126338

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 24 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l1263_126338


namespace NUMINAMATH_CALUDE_cos_15_degrees_l1263_126397

theorem cos_15_degrees : Real.cos (15 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_15_degrees_l1263_126397


namespace NUMINAMATH_CALUDE_segments_between_five_points_segments_between_five_points_proof_l1263_126326

/-- Given 5 points where no three are collinear, the number of segments needed to connect each pair of points is 10. -/
theorem segments_between_five_points : ℕ → Prop :=
  fun n => n = 5 → (∀ p1 p2 p3 : ℝ × ℝ, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬Collinear p1 p2 p3) →
    (Nat.choose n 2 = 10)
  where
    Collinear (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) (p3 : ℝ × ℝ) : Prop :=
      ∃ t : ℝ, p3 = p1 + t • (p2 - p1)

/-- Proof of the theorem -/
theorem segments_between_five_points_proof : segments_between_five_points 5 := by
  sorry

end NUMINAMATH_CALUDE_segments_between_five_points_segments_between_five_points_proof_l1263_126326


namespace NUMINAMATH_CALUDE_candied_yams_ratio_l1263_126377

/-- The ratio of shoppers who buy candied yams to the total number of shoppers -/
theorem candied_yams_ratio 
  (packages_per_box : ℕ) 
  (boxes_ordered : ℕ) 
  (total_shoppers : ℕ) 
  (h1 : packages_per_box = 25)
  (h2 : boxes_ordered = 5)
  (h3 : total_shoppers = 375) : 
  (boxes_ordered * packages_per_box : ℚ) / total_shoppers = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_candied_yams_ratio_l1263_126377


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l1263_126327

/-- Represents a 2D point -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line in parametric form -/
structure ParametricLine where
  p : Point  -- Point on the line
  v : Point  -- Direction vector

/-- The first line -/
def line1 : ParametricLine :=
  { p := { x := 2, y := 2 },
    v := { x := 3, y := -4 } }

/-- The second line -/
def line2 : ParametricLine :=
  { p := { x := 7, y := -6 },
    v := { x := 5, y := 3 } }

/-- The claimed intersection point -/
def intersectionPoint : Point :=
  { x := 11, y := -886/87 }

/-- Theorem stating that the given point is the unique intersection of the two lines -/
theorem intersection_point_is_unique :
  ∃! t u : ℚ,
    line1.p.x + t * line1.v.x = intersectionPoint.x ∧
    line1.p.y + t * line1.v.y = intersectionPoint.y ∧
    line2.p.x + u * line2.v.x = intersectionPoint.x ∧
    line2.p.y + u * line2.v.y = intersectionPoint.y :=
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l1263_126327


namespace NUMINAMATH_CALUDE_carls_playground_area_l1263_126350

/-- Represents a rectangular playground with fence posts. -/
structure Playground where
  total_posts : ℕ
  post_spacing : ℝ
  short_side_posts : ℕ
  long_side_posts : ℕ

/-- Calculates the area of the playground given its specifications. -/
def calculate_area (p : Playground) : ℝ :=
  ((p.short_side_posts - 1) * p.post_spacing) * ((p.long_side_posts - 1) * p.post_spacing)

/-- Theorem stating the area of Carl's playground is 324 square yards. -/
theorem carls_playground_area :
  ∃ (p : Playground),
    p.total_posts = 24 ∧
    p.post_spacing = 3 ∧
    p.long_side_posts = 2 * p.short_side_posts ∧
    calculate_area p = 324 := by
  sorry

end NUMINAMATH_CALUDE_carls_playground_area_l1263_126350


namespace NUMINAMATH_CALUDE_apples_needed_for_pies_l1263_126382

theorem apples_needed_for_pies (pies_to_bake : ℕ) (apples_per_pie : ℕ) (apples_on_hand : ℕ) : 
  pies_to_bake * apples_per_pie - apples_on_hand = 110 :=
by
  sorry

#check apples_needed_for_pies 15 10 40

end NUMINAMATH_CALUDE_apples_needed_for_pies_l1263_126382


namespace NUMINAMATH_CALUDE_sum_difference_equals_three_l1263_126372

theorem sum_difference_equals_three : (2 + 4 + 6) - (1 + 3 + 5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_equals_three_l1263_126372


namespace NUMINAMATH_CALUDE_bucket_capacity_first_case_l1263_126300

/-- The capacity of a bucket in the first case, given the following conditions:
  - 22 buckets of water fill a tank in the first case
  - 33 buckets of water fill the same tank in the second case
  - In the second case, each bucket has a capacity of 9 litres
-/
theorem bucket_capacity_first_case : 
  ∀ (capacity_first : ℝ) (tank_volume : ℝ),
  22 * capacity_first = tank_volume →
  33 * 9 = tank_volume →
  capacity_first = 13.5 := by
sorry

end NUMINAMATH_CALUDE_bucket_capacity_first_case_l1263_126300


namespace NUMINAMATH_CALUDE_intersection_complement_when_m_is_one_union_equals_A_iff_m_in_range_l1263_126385

-- Define sets A and B
def A : Set ℝ := {x | x^2 + 5*x - 6 < 0}
def B (m : ℝ) : Set ℝ := {x | m - 2 < x ∧ x < 2*m + 1}

-- Part 1
theorem intersection_complement_when_m_is_one :
  A ∩ (Set.univ \ B 1) = {x | -6 < x ∧ x ≤ -1} := by sorry

-- Part 2
theorem union_equals_A_iff_m_in_range :
  ∀ m : ℝ, A ∪ B m = A ↔ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_m_is_one_union_equals_A_iff_m_in_range_l1263_126385


namespace NUMINAMATH_CALUDE_donovans_test_score_l1263_126367

theorem donovans_test_score (incorrect_answers : ℕ) (correct_percentage : ℚ) 
  (h1 : incorrect_answers = 13)
  (h2 : correct_percentage = 7292 / 10000) : 
  ∃ (correct_answers : ℕ), 
    (correct_answers : ℚ) / ((correct_answers : ℚ) + (incorrect_answers : ℚ)) = correct_percentage ∧ 
    correct_answers = 35 := by
  sorry

end NUMINAMATH_CALUDE_donovans_test_score_l1263_126367


namespace NUMINAMATH_CALUDE_product_evaluation_l1263_126363

theorem product_evaluation :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) * (3^8 + 1^8) = 21523360 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l1263_126363


namespace NUMINAMATH_CALUDE_complex_on_real_axis_l1263_126334

theorem complex_on_real_axis (a : ℝ) : 
  let z : ℂ := (a - Complex.I) * (1 + Complex.I)
  (z.im = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_real_axis_l1263_126334


namespace NUMINAMATH_CALUDE_line_separate_from_circle_l1263_126383

/-- A line with negative slope intersecting a circle is separate from another circle -/
theorem line_separate_from_circle (k : ℝ) (h_k : k < 0) : 
  ∃ (x y : ℝ), y = k * x ∧ (x + 3)^2 + (y + 2)^2 = 9 →
  ∀ (x y : ℝ), y = k * x → x^2 + (y - 2)^2 > 1 := by
sorry

end NUMINAMATH_CALUDE_line_separate_from_circle_l1263_126383


namespace NUMINAMATH_CALUDE_specific_tree_height_l1263_126398

/-- Represents the height of a tree after a given number of years -/
def tree_height (initial_height : ℝ) (yearly_growth : ℝ) (years : ℝ) : ℝ :=
  initial_height + yearly_growth * years

/-- Theorem stating the height of a specific tree after n years -/
theorem specific_tree_height (n : ℝ) :
  tree_height 1.8 0.3 n = 0.3 * n + 1.8 := by
  sorry

end NUMINAMATH_CALUDE_specific_tree_height_l1263_126398


namespace NUMINAMATH_CALUDE_urn_ball_removal_l1263_126304

theorem urn_ball_removal (total : ℕ) (red_percent : ℚ) (blue_removed : ℕ) (new_red_percent : ℚ) : 
  total = 150 →
  red_percent = 2/5 →
  blue_removed = 75 →
  new_red_percent = 4/5 →
  (red_percent * total : ℚ) / (total - blue_removed : ℚ) = new_red_percent :=
by sorry

end NUMINAMATH_CALUDE_urn_ball_removal_l1263_126304


namespace NUMINAMATH_CALUDE_investment_growth_l1263_126331

/-- Represents the investment growth over a two-year period -/
theorem investment_growth 
  (initial_investment : ℝ) 
  (final_investment : ℝ) 
  (growth_rate : ℝ) 
  (h1 : initial_investment = 1500)
  (h2 : final_investment = 4250)
  (h3 : initial_investment * (1 + growth_rate)^2 = final_investment) :
  1500 * (1 + growth_rate)^2 = 4250 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l1263_126331


namespace NUMINAMATH_CALUDE_expression_value_l1263_126308

theorem expression_value (x : ℝ) (h : x^2 - 5*x - 2006 = 0) :
  ((x-2)^3 - (x-1)^2 + 1) / (x-2) = 2010 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1263_126308


namespace NUMINAMATH_CALUDE_solve_barnyard_owl_problem_l1263_126384

def barnyard_owl_problem (hoots_per_owl : ℕ) (total_hoots : ℕ) : Prop :=
  let num_owls := (20 - 5) / hoots_per_owl
  hoots_per_owl = 5 ∧ total_hoots = 20 - 5 → num_owls = 3

theorem solve_barnyard_owl_problem :
  ∃ (hoots_per_owl total_hoots : ℕ), barnyard_owl_problem hoots_per_owl total_hoots :=
sorry

end NUMINAMATH_CALUDE_solve_barnyard_owl_problem_l1263_126384


namespace NUMINAMATH_CALUDE_transmission_time_is_8_67_minutes_l1263_126325

/-- Represents the number of chunks in a regular block -/
def regular_block_chunks : ℕ := 800

/-- Represents the number of chunks in a large block -/
def large_block_chunks : ℕ := 1600

/-- Represents the number of regular blocks -/
def num_regular_blocks : ℕ := 70

/-- Represents the number of large blocks -/
def num_large_blocks : ℕ := 30

/-- Represents the transmission rate in chunks per second -/
def transmission_rate : ℕ := 200

/-- Calculates the total number of chunks to be transmitted -/
def total_chunks : ℕ := 
  num_regular_blocks * regular_block_chunks + num_large_blocks * large_block_chunks

/-- Calculates the transmission time in seconds -/
def transmission_time_seconds : ℕ := total_chunks / transmission_rate

/-- Theorem stating that the transmission time is 8.67 minutes -/
theorem transmission_time_is_8_67_minutes : 
  (transmission_time_seconds : ℚ) / 60 = 8.67 := by sorry

end NUMINAMATH_CALUDE_transmission_time_is_8_67_minutes_l1263_126325


namespace NUMINAMATH_CALUDE_jane_work_days_jane_solo_days_l1263_126322

theorem jane_work_days (john_days : ℝ) (total_days : ℝ) (jane_stop_days : ℝ) : ℝ :=
  let john_rate := 1 / john_days
  let total_work := 1
  let jane_work_days := total_days - jane_stop_days
  let john_solo_work := john_rate * jane_stop_days
  let combined_work := total_work - john_solo_work
  combined_work / (john_rate + 1 / (total_days - jane_stop_days)) / jane_work_days

theorem jane_solo_days 
  (john_days : ℝ) 
  (total_days : ℝ) 
  (jane_stop_days : ℝ) 
  (h1 : john_days = 20)
  (h2 : total_days = 10)
  (h3 : jane_stop_days = 4)
  : jane_work_days john_days total_days jane_stop_days = 12 := by
  sorry

end NUMINAMATH_CALUDE_jane_work_days_jane_solo_days_l1263_126322


namespace NUMINAMATH_CALUDE_triangle_k_range_l1263_126359

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three lines form a triangle -/
def form_triangle (l₁ l₂ l₃ : Line) : Prop :=
  ∀ (x y : ℝ), (l₁.a * x + l₁.b * y + l₁.c = 0 ∧ 
                l₂.a * x + l₂.b * y + l₂.c = 0 ∧
                l₃.a * x + l₃.b * y + l₃.c = 0) → False

/-- The theorem stating the range of k for which the given lines form a triangle -/
theorem triangle_k_range :
  ∀ (k : ℝ),
  let l₁ : Line := ⟨1, -1, 0⟩
  let l₂ : Line := ⟨1, 1, -2⟩
  let l₃ : Line := ⟨5, -k, -15⟩
  form_triangle l₁ l₂ l₃ ↔ k ≠ 5 ∧ k ≠ -5 ∧ k ≠ -10 :=
sorry

end NUMINAMATH_CALUDE_triangle_k_range_l1263_126359


namespace NUMINAMATH_CALUDE_fencing_cost_theorem_l1263_126351

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length : ℝ) (breadth : ℝ) (cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_meter

/-- Theorem: The total cost of fencing the given rectangular plot is 5300 currency units -/
theorem fencing_cost_theorem :
  let length : ℝ := 63
  let breadth : ℝ := 37
  let cost_per_meter : ℝ := 26.50
  total_fencing_cost length breadth cost_per_meter = 5300 := by
  sorry

#eval total_fencing_cost 63 37 26.50

end NUMINAMATH_CALUDE_fencing_cost_theorem_l1263_126351


namespace NUMINAMATH_CALUDE_sequence_integer_count_l1263_126358

def sequence_term (n : ℕ) : ℚ :=
  8820 / 3^n

def is_integer (q : ℚ) : Prop :=
  ∃ z : ℤ, q = z

theorem sequence_integer_count :
  (∃ n : ℕ, n > 0 ∧ 
    (∀ k < n, is_integer (sequence_term k)) ∧
    ¬is_integer (sequence_term n)) ∧
  (∀ m : ℕ, m > 0 →
    (∀ k < m, is_integer (sequence_term k)) →
    ¬is_integer (sequence_term m) →
    m = 3) :=
sorry

end NUMINAMATH_CALUDE_sequence_integer_count_l1263_126358


namespace NUMINAMATH_CALUDE_hidden_dots_count_l1263_126302

/-- Represents a standard six-sided die -/
def standardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The sum of all numbers on a standard die -/
def standardDieSum : ℕ := Finset.sum standardDie id

/-- The number of dice in the stack -/
def numDice : ℕ := 4

/-- The visible numbers on the dice -/
def visibleNumbers : Finset ℕ := {1, 2, 2, 3, 3, 4, 5, 6}

/-- The sum of visible numbers -/
def visibleSum : ℕ := Finset.sum visibleNumbers id

/-- The total number of dots on all dice -/
def totalDots : ℕ := numDice * standardDieSum

theorem hidden_dots_count : totalDots - visibleSum = 58 := by sorry

end NUMINAMATH_CALUDE_hidden_dots_count_l1263_126302


namespace NUMINAMATH_CALUDE_chess_match_schedules_count_l1263_126389

/-- Represents a chess match schedule between two schools -/
structure ChessMatchSchedule where
  /-- Number of players per school -/
  players_per_school : Nat
  /-- Number of games played simultaneously in each round -/
  games_per_round : Nat
  /-- Total number of games in the match -/
  total_games : Nat
  /-- Number of rounds in the match -/
  total_rounds : Nat
  /-- Condition: Each player plays against each player from the other school -/
  player_matchup : players_per_school * players_per_school = total_games
  /-- Condition: Games are evenly distributed across rounds -/
  round_distribution : total_games = games_per_round * total_rounds

/-- The number of different ways to schedule the chess match -/
def number_of_schedules (schedule : ChessMatchSchedule) : Nat :=
  Nat.factorial schedule.total_rounds

/-- Theorem stating that there are 24 different ways to schedule the chess match -/
theorem chess_match_schedules_count :
  ∃ (schedule : ChessMatchSchedule),
    schedule.players_per_school = 4 ∧
    schedule.games_per_round = 4 ∧
    number_of_schedules schedule = 24 := by
  sorry

end NUMINAMATH_CALUDE_chess_match_schedules_count_l1263_126389


namespace NUMINAMATH_CALUDE_sum_of_square_roots_inequality_l1263_126307

theorem sum_of_square_roots_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab_c : a + b + c = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_inequality_l1263_126307


namespace NUMINAMATH_CALUDE_automobile_distance_l1263_126329

/-- 
Given an automobile that travels 2a/5 feet in r seconds, 
this theorem proves that it will travel 40a/r yards in 5 minutes 
if this rate is maintained.
-/
theorem automobile_distance (a r : ℝ) (hr : r > 0) : 
  let rate_feet_per_second := (2 * a / 5) / r
  let rate_yards_per_second := rate_feet_per_second / 3
  let time_in_seconds := 5 * 60
  rate_yards_per_second * time_in_seconds = 40 * a / r := by
  sorry

end NUMINAMATH_CALUDE_automobile_distance_l1263_126329


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_with_8_and_0_l1263_126318

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 8 ∨ d = 0

def count_digit (n : ℕ) (d : ℕ) : ℕ :=
  (n.digits 10).filter (· = d) |>.length

theorem largest_multiple_of_15_with_8_and_0 :
  ∃ m : ℕ,
    m > 0 ∧
    15 ∣ m ∧
    is_valid_number m ∧
    count_digit m 8 = 6 ∧
    count_digit m 0 = 1 ∧
    m / 15 = 592592 ∧
    ∀ n : ℕ, n > m → ¬(15 ∣ n ∧ is_valid_number n) :=
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_with_8_and_0_l1263_126318


namespace NUMINAMATH_CALUDE_triangle_count_is_nine_l1263_126375

/-- Represents the triangular grid structure described in the problem -/
structure TriangularGrid :=
  (top_row : Nat)
  (middle_row : Nat)
  (bottom_row : Nat)
  (has_inverted_triangle : Bool)

/-- Calculates the total number of triangles in the given grid -/
def count_triangles (grid : TriangularGrid) : Nat :=
  let small_triangles := grid.top_row + grid.middle_row + grid.bottom_row
  let medium_triangles := if grid.top_row ≥ 3 then 1 else 0 +
                          if grid.middle_row + grid.bottom_row ≥ 3 then 1 else 0
  let large_triangle := if grid.has_inverted_triangle then 1 else 0
  small_triangles + medium_triangles + large_triangle

/-- The specific grid described in the problem -/
def problem_grid : TriangularGrid :=
  { top_row := 3,
    middle_row := 2,
    bottom_row := 1,
    has_inverted_triangle := true }

theorem triangle_count_is_nine :
  count_triangles problem_grid = 9 :=
sorry

end NUMINAMATH_CALUDE_triangle_count_is_nine_l1263_126375


namespace NUMINAMATH_CALUDE_expression_equality_l1263_126346

theorem expression_equality : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1263_126346


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_is_9_6_l1263_126373

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : std_dev > 0

/-- The value that is exactly n standard deviations less than the mean -/
def value_n_std_dev_below_mean (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- Theorem: For a normal distribution with mean 12 and standard deviation 1.2,
    the value that is exactly 2 standard deviations less than the mean is 9.6 -/
theorem two_std_dev_below_mean_is_9_6 :
  let d : NormalDistribution := ⟨12, 1.2, by norm_num⟩
  value_n_std_dev_below_mean d 2 = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_is_9_6_l1263_126373


namespace NUMINAMATH_CALUDE_teacher_zhang_age_in_five_years_l1263_126343

/-- Given Xiao Li's age and the relationship between Xiao Li's and Teacher Zhang's ages,
    prove Teacher Zhang's age after 5 years. -/
theorem teacher_zhang_age_in_five_years (a : ℕ) : 
  (3 * a - 2) + 5 = 3 * a + 3 :=
by sorry

end NUMINAMATH_CALUDE_teacher_zhang_age_in_five_years_l1263_126343


namespace NUMINAMATH_CALUDE_toothpicks_stage_15_l1263_126335

/-- Calculates the number of toothpicks at a given stage -/
def toothpicks (stage : ℕ) : ℕ :=
  let initial := 3
  let baseIncrease := 2
  let extraIncreaseInterval := 3
  let extraIncrease := (stage - 1) / extraIncreaseInterval

  initial + (stage - 1) * baseIncrease + 
    ((stage - 1) / extraIncreaseInterval) * (stage - 1) * (stage - 2) / 2

theorem toothpicks_stage_15 : toothpicks 15 = 61 := by
  sorry

#eval toothpicks 15

end NUMINAMATH_CALUDE_toothpicks_stage_15_l1263_126335


namespace NUMINAMATH_CALUDE_broken_line_isoperimetric_inequality_l1263_126340

/-- A non-self-intersecting broken line in a half-plane -/
structure BrokenLine where
  length : ℝ
  area : ℝ
  nonSelfIntersecting : Prop
  endsOnBoundary : Prop

/-- The isoperimetric inequality for the broken line -/
theorem broken_line_isoperimetric_inequality (b : BrokenLine) :
  b.area ≤ b.length^2 / (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_broken_line_isoperimetric_inequality_l1263_126340


namespace NUMINAMATH_CALUDE_ohara_triple_49_16_l1263_126362

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ x > 0 ∧ Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x

/-- Theorem: If (49, 16, x) is an O'Hara triple, then x = 11 -/
theorem ohara_triple_49_16 (x : ℕ) :
  is_ohara_triple 49 16 x → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_49_16_l1263_126362


namespace NUMINAMATH_CALUDE_peter_pizza_fraction_l1263_126394

theorem peter_pizza_fraction :
  ∀ (total_slices : ℕ) (peter_own_slices : ℕ) (shared_slices : ℕ),
  total_slices = 16 →
  peter_own_slices = 2 →
  shared_slices = 2 →
  (peter_own_slices : ℚ) / total_slices + (shared_slices / 2 : ℚ) / total_slices = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_peter_pizza_fraction_l1263_126394


namespace NUMINAMATH_CALUDE_space_shuttle_speed_l1263_126355

-- Define the speed in kilometers per hour
def speed_kmh : ℝ := 14400

-- Define the conversion factor from hours to seconds
def seconds_per_hour : ℝ := 3600

-- The theorem to prove
theorem space_shuttle_speed : speed_kmh / seconds_per_hour = 4 := by
  sorry

end NUMINAMATH_CALUDE_space_shuttle_speed_l1263_126355


namespace NUMINAMATH_CALUDE_find_number_l1263_126312

theorem find_number : ∃ x : ℕ,
  x % 18 = 6 ∧
  190 % 18 = 10 ∧
  x < 190 ∧
  (∀ y : ℕ, y % 18 = 6 → y < 190 → y ≤ x) ∧
  x = 186 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1263_126312


namespace NUMINAMATH_CALUDE_average_glasses_per_box_l1263_126315

/-- Proves that given the specified conditions, the average number of glasses per box is 15 -/
theorem average_glasses_per_box : 
  ∀ (small_boxes large_boxes : ℕ),
  small_boxes > 0 →
  large_boxes = small_boxes + 16 →
  12 * small_boxes + 16 * large_boxes = 480 →
  (480 : ℚ) / (small_boxes + large_boxes) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_average_glasses_per_box_l1263_126315


namespace NUMINAMATH_CALUDE_total_eyes_in_pond_l1263_126354

/-- The number of snakes in the pond -/
def num_snakes : ℕ := 18

/-- The number of alligators in the pond -/
def num_alligators : ℕ := 10

/-- The number of eyes each snake has -/
def eyes_per_snake : ℕ := 2

/-- The number of eyes each alligator has -/
def eyes_per_alligator : ℕ := 2

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := num_snakes * eyes_per_snake + num_alligators * eyes_per_alligator

theorem total_eyes_in_pond : total_eyes = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_eyes_in_pond_l1263_126354


namespace NUMINAMATH_CALUDE_tv_show_average_episodes_l1263_126323

theorem tv_show_average_episodes (total_years : ℕ) (seasons_15 : ℕ) (seasons_20 : ℕ) (seasons_12 : ℕ)
  (h1 : total_years = 14)
  (h2 : seasons_15 = 8)
  (h3 : seasons_20 = 4)
  (h4 : seasons_12 = 2) :
  (seasons_15 * 15 + seasons_20 * 20 + seasons_12 * 12) / total_years = 16 := by
  sorry

end NUMINAMATH_CALUDE_tv_show_average_episodes_l1263_126323


namespace NUMINAMATH_CALUDE_f_equals_g_l1263_126353

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 1
def g (t : ℝ) : ℝ := t^2 + 1

-- Theorem stating that f and g represent the same function
theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l1263_126353


namespace NUMINAMATH_CALUDE_expand_expression_l1263_126391

theorem expand_expression (x : ℝ) : (17*x + 18 + 5)*3*x = 51*x^2 + 69*x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1263_126391


namespace NUMINAMATH_CALUDE_inequality_representation_l1263_126356

/-- 
Theorem: The inequality 3x - 2 > 0 correctly represents the statement 
"x is three times the difference between 2".
-/
theorem inequality_representation (x : ℝ) : 
  (3 * x - 2 > 0) ↔ (∃ y : ℝ, x = 3 * y ∧ y > 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_representation_l1263_126356


namespace NUMINAMATH_CALUDE_triangle_properties_l1263_126369

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- Theorem statement
theorem triangle_properties (t : AcuteTriangle) 
  (h1 : t.c = 2) 
  (h2 : t.A = π/3) : 
  t.a * Real.sin t.C = Real.sqrt 3 ∧ 
  1 + Real.sqrt 3 < t.a + t.b ∧ 
  t.a + t.b < 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1263_126369


namespace NUMINAMATH_CALUDE_sqrt_square_abs_two_div_sqrt_two_l1263_126311

-- Theorem 1: For any real number x, sqrt(x^2) = |x|
theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

-- Theorem 2: 2 / sqrt(2) = sqrt(2)
theorem two_div_sqrt_two : 2 / Real.sqrt 2 = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_abs_two_div_sqrt_two_l1263_126311


namespace NUMINAMATH_CALUDE_coffee_cost_l1263_126395

/-- The cost of each coffee Jon buys, given his spending habits in April. -/
theorem coffee_cost (coffees_per_day : ℕ) (total_spent : ℕ) (days_in_april : ℕ) :
  coffees_per_day = 2 →
  total_spent = 120 →
  days_in_april = 30 →
  total_spent / (coffees_per_day * days_in_april) = 2 :=
by sorry

end NUMINAMATH_CALUDE_coffee_cost_l1263_126395


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1263_126387

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1263_126387


namespace NUMINAMATH_CALUDE_divisors_sum_and_product_l1263_126378

theorem divisors_sum_and_product (p : ℕ) (hp : Prime p) :
  let a := p^106
  ∀ (divisors : Finset ℕ), 
    (∀ d ∈ divisors, d ∣ a) ∧ 
    (∀ d : ℕ, d ∣ a → d ∈ divisors) ∧ 
    (Finset.card divisors = 107) →
    (divisors.sum id = (p^107 - 1) / (p - 1)) ∧
    (divisors.prod id = p^11321) := by
  sorry

end NUMINAMATH_CALUDE_divisors_sum_and_product_l1263_126378


namespace NUMINAMATH_CALUDE_derivative_y_l1263_126337

noncomputable def y (x : ℝ) : ℝ := x * Real.sin (2 * x)

theorem derivative_y (x : ℝ) :
  deriv y x = Real.sin (2 * x) + 2 * x * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_y_l1263_126337


namespace NUMINAMATH_CALUDE_mikes_purchase_cost_l1263_126339

/-- The total cost of a camera and lens purchase -/
def total_cost (old_camera_cost lens_price lens_discount : ℚ) : ℚ :=
  let new_camera_cost := old_camera_cost * (1 + 0.3)
  let discounted_lens_price := lens_price - lens_discount
  new_camera_cost + discounted_lens_price

/-- Theorem stating the total cost of Mike's camera and lens purchase -/
theorem mikes_purchase_cost :
  total_cost 4000 400 200 = 5400 := by
  sorry

end NUMINAMATH_CALUDE_mikes_purchase_cost_l1263_126339


namespace NUMINAMATH_CALUDE_linear_function_property_l1263_126319

/-- A linear function is a function of the form f(x) = mx + b, where m and b are constants. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

theorem linear_function_property (g : ℝ → ℝ) 
  (hLinear : LinearFunction g) 
  (hDiff : g 10 - g 0 = 20) : 
  g 20 - g 0 = 40 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l1263_126319


namespace NUMINAMATH_CALUDE_petrol_price_equation_l1263_126316

/-- The original price of petrol per gallon -/
def P : ℝ := sorry

/-- The reduced price is 90% of the original price -/
def reduced_price : ℝ := 0.9 * P

/-- The equation representing the relationship between the original and reduced prices -/
theorem petrol_price_equation : 250 / reduced_price = 250 / P + 5 := by sorry

end NUMINAMATH_CALUDE_petrol_price_equation_l1263_126316


namespace NUMINAMATH_CALUDE_prob_diamond_or_club_half_l1263_126333

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (cards_per_suit : ℕ)
  (diamond_club_cards : ℕ)

/-- Probability of drawing a diamond or club from the top of a shuffled deck -/
def prob_diamond_or_club (d : Deck) : ℚ :=
  d.diamond_club_cards / d.total_cards

/-- Theorem stating the probability of drawing a diamond or club is 1/2 -/
theorem prob_diamond_or_club_half (d : Deck) 
  (h1 : d.total_cards = 52) 
  (h2 : d.cards_per_suit = 13) 
  (h3 : d.diamond_club_cards = 2 * d.cards_per_suit) : 
  prob_diamond_or_club d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_diamond_or_club_half_l1263_126333


namespace NUMINAMATH_CALUDE_collinear_points_a_equals_4_l1263_126341

/-- Three points are collinear if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- Given three points A(a,2), B(5,1), and C(-4,2a) are collinear, prove that a = 4. -/
theorem collinear_points_a_equals_4 (a : ℝ) :
  collinear a 2 5 1 (-4) (2*a) → a = 4 := by
  sorry


end NUMINAMATH_CALUDE_collinear_points_a_equals_4_l1263_126341


namespace NUMINAMATH_CALUDE_total_pens_count_l1263_126317

/-- The number of black pens bought by the teacher -/
def black_pens : ℕ := 7

/-- The number of blue pens bought by the teacher -/
def blue_pens : ℕ := 9

/-- The number of red pens bought by the teacher -/
def red_pens : ℕ := 5

/-- The total number of pens bought by the teacher -/
def total_pens : ℕ := black_pens + blue_pens + red_pens

theorem total_pens_count : total_pens = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_pens_count_l1263_126317


namespace NUMINAMATH_CALUDE_unique_divisor_sums_l1263_126352

def divisor_sums (n : ℕ+) : Finset ℕ :=
  (Finset.powerset (Nat.divisors n.val)).image (λ s => s.sum id)

def target_sums : Finset ℕ := {4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 46, 48, 50, 54, 60}

theorem unique_divisor_sums (n : ℕ+) : divisor_sums n = target_sums → n = 45 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_sums_l1263_126352


namespace NUMINAMATH_CALUDE_range_of_a_l1263_126344

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then x + 4 else x^2 - 2*x

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → a ∈ Set.Icc (-5 : ℝ) (4 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1263_126344


namespace NUMINAMATH_CALUDE_walter_at_zoo_l1263_126399

theorem walter_at_zoo (seal_time penguin_time elephant_time total_time : ℕ) 
  (h1 : penguin_time = 8 * seal_time)
  (h2 : elephant_time = 13)
  (h3 : total_time = 130)
  (h4 : seal_time + penguin_time + elephant_time = total_time) :
  seal_time = 13 := by
  sorry

end NUMINAMATH_CALUDE_walter_at_zoo_l1263_126399


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l1263_126320

theorem smallest_number_of_eggs (total_containers : ℕ) (deficient_containers : ℕ) 
  (container_capacity : ℕ) (min_total_eggs : ℕ) :
  total_containers > 10 ∧ 
  deficient_containers = 3 ∧ 
  container_capacity = 15 ∧ 
  min_total_eggs = 150 →
  (total_containers * container_capacity - deficient_containers = 
    (total_containers - deficient_containers) * container_capacity + 
    deficient_containers * (container_capacity - 1)) ∧
  (total_containers * container_capacity - deficient_containers > min_total_eggs) ∧
  ∀ n : ℕ, n < total_containers → 
    n * container_capacity - deficient_containers ≤ min_total_eggs :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l1263_126320


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_2_range_of_a_for_solutions_l1263_126379

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 1|

-- Theorem for the solution set of f(x) < 2
theorem solution_set_f_less_than_2 :
  {x : ℝ | f x < 2} = Set.Ioo (-4 : ℝ) (2/3) := by sorry

-- Theorem for the range of a
theorem range_of_a_for_solutions (a : ℝ) :
  (∃ x, f x ≤ a - a^2/2) ↔ a ∈ Set.Icc (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_2_range_of_a_for_solutions_l1263_126379


namespace NUMINAMATH_CALUDE_product_sum_relation_l1263_126376

theorem product_sum_relation (a b : ℝ) : 
  (a * b = 2 * (a + b) + 12) → (b = 10) → (b - a = 6) := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l1263_126376


namespace NUMINAMATH_CALUDE_lcm_of_153_180_560_l1263_126303

theorem lcm_of_153_180_560 : Nat.lcm 153 (Nat.lcm 180 560) = 85680 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_153_180_560_l1263_126303


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l1263_126324

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  tens_nonzero : 1 ≤ tens ∧ tens ≤ 9
  units_bound : units ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- The reverse of a two-digit number -/
def TwoDigitNumber.reverse (n : TwoDigitNumber) : TwoDigitNumber where
  tens := n.units
  units := n.tens
  tens_nonzero := by sorry
  units_bound := n.tens_nonzero.2

theorem two_digit_number_sum (n : TwoDigitNumber) :
  (n.value - n.reverse.value = 7 * (n.tens + n.units)) →
  (n.value + n.reverse.value = 99) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l1263_126324


namespace NUMINAMATH_CALUDE_product_remainder_zero_l1263_126370

theorem product_remainder_zero : (4251 * 7396 * 4625) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_zero_l1263_126370


namespace NUMINAMATH_CALUDE_square_area_is_25_l1263_126310

-- Define the points
def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (5, 6)

-- Define the square area function
def square_area (p1 p2 : ℝ × ℝ) : ℝ :=
  let dx := p2.1 - p1.1
  let dy := p2.2 - p1.2
  (dx * dx + dy * dy)

-- Theorem statement
theorem square_area_is_25 : square_area point1 point2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_25_l1263_126310


namespace NUMINAMATH_CALUDE_cubic_factorization_l1263_126361

theorem cubic_factorization (a : ℝ) : a^3 - 16*a = a*(a+4)*(a-4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1263_126361


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l1263_126305

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The given complex number -/
def z : ℂ := (1 + 2 * i) * i

/-- A complex number is in the second quadrant if its real part is negative and its imaginary part is positive -/
def is_in_second_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im > 0

theorem z_in_second_quadrant : is_in_second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l1263_126305


namespace NUMINAMATH_CALUDE_x_equals_y_plus_m_percent_l1263_126374

-- Define the relationship between x, y, and m
def is_m_percent_more (x y m : ℝ) : Prop :=
  x = y + (m / 100) * y

-- Theorem statement
theorem x_equals_y_plus_m_percent (x y m : ℝ) :
  is_m_percent_more x y m → x = (100 + m) / 100 * y := by
  sorry

end NUMINAMATH_CALUDE_x_equals_y_plus_m_percent_l1263_126374


namespace NUMINAMATH_CALUDE_exists_f_1984_eq_A_l1263_126321

-- Define the function property
def satisfies_property (f : ℤ → ℝ) : Prop :=
  ∀ x y : ℤ, f (x - y^2) = f x + (y^2 - 2*x) * f y

-- State the theorem
theorem exists_f_1984_eq_A (A : ℝ) :
  ∃ f : ℤ → ℝ, satisfies_property f ∧ f 1984 = A :=
sorry

end NUMINAMATH_CALUDE_exists_f_1984_eq_A_l1263_126321


namespace NUMINAMATH_CALUDE_brother_contribution_l1263_126314

/-- The number of wood pieces Alvin needs in total -/
def total_needed : ℕ := 376

/-- The number of wood pieces Alvin's friend gave him -/
def friend_gave : ℕ := 123

/-- The number of wood pieces Alvin still needs to gather -/
def still_needed : ℕ := 117

/-- The number of wood pieces Alvin's brother gave him -/
def brother_gave : ℕ := total_needed - friend_gave - still_needed

theorem brother_contribution : brother_gave = 136 := by
  sorry

end NUMINAMATH_CALUDE_brother_contribution_l1263_126314


namespace NUMINAMATH_CALUDE_four_color_plane_partition_l1263_126328

-- Define the plane as ℝ × ℝ
def Plane := ℝ × ℝ

-- Define a partition of the plane into four subsets
def Partition (A B C D : Set Plane) : Prop :=
  (A ∪ B ∪ C ∪ D = Set.univ) ∧
  (A ∩ B = ∅) ∧ (A ∩ C = ∅) ∧ (A ∩ D = ∅) ∧
  (B ∩ C = ∅) ∧ (B ∩ D = ∅) ∧ (C ∩ D = ∅)

-- Define a circle in the plane
def Circle (center : Plane) (radius : ℝ) : Set Plane :=
  {p : Plane | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Theorem statement
theorem four_color_plane_partition :
  ∃ (A B C D : Set Plane), Partition A B C D ∧
    ∀ (center : Plane) (radius : ℝ),
      (Circle center radius ∩ A).Nonempty ∧
      (Circle center radius ∩ B).Nonempty ∧
      (Circle center radius ∩ C).Nonempty ∧
      (Circle center radius ∩ D).Nonempty :=
by sorry


end NUMINAMATH_CALUDE_four_color_plane_partition_l1263_126328


namespace NUMINAMATH_CALUDE_train_crossing_time_l1263_126357

/-- Proves that a train with given specifications takes 20 seconds to cross a platform -/
theorem train_crossing_time (train_length : ℝ) (platform_length : ℝ) (passing_time : ℝ) :
  train_length = 180 →
  platform_length = 270 →
  passing_time = 8 →
  let train_speed := train_length / passing_time
  let total_distance := train_length + platform_length
  let crossing_time := total_distance / train_speed
  crossing_time = 20 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1263_126357


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l1263_126345

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l1263_126345


namespace NUMINAMATH_CALUDE_problem_statement_l1263_126348

theorem problem_statement : 65 * 1515 - 25 * 1515 + 1515 = 62115 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1263_126348


namespace NUMINAMATH_CALUDE_inequality_proof_l1263_126365

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) + Real.sqrt ((a * b + b * c + c * a) / (a^2 + b^2 + c^2)) ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1263_126365


namespace NUMINAMATH_CALUDE_hospital_current_age_l1263_126309

/-- Represents the current age of Grant -/
def grants_current_age : ℕ := 25

/-- Represents the number of years in the future when the condition is met -/
def years_in_future : ℕ := 5

/-- Represents the fraction of the hospital's age that Grant will be in the future -/
def age_fraction : ℚ := 2/3

/-- Theorem stating that given the conditions, the current age of the hospital is 40 years -/
theorem hospital_current_age : 
  ∃ (hospital_age : ℕ), 
    (grants_current_age + years_in_future : ℚ) = age_fraction * (hospital_age + years_in_future : ℚ) ∧
    hospital_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_hospital_current_age_l1263_126309


namespace NUMINAMATH_CALUDE_sum_of_factors_l1263_126371

theorem sum_of_factors (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e →
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 0 →
  a + b + c + d + e = 35 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l1263_126371


namespace NUMINAMATH_CALUDE_determine_origin_l1263_126332

/-- Given two points A and B in a 2D coordinate system, we can uniquely determine the origin. -/
theorem determine_origin (A B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (3, 1)) :
  ∃! O : ℝ × ℝ, O = (0, 0) ∧ 
  (O.1 - A.1) ^ 2 + (O.2 - A.2) ^ 2 = (O.1 - B.1) ^ 2 + (O.2 - B.2) ^ 2 ∧
  (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = (O.1 - A.1) ^ 2 + (O.2 - A.2) ^ 2 + (O.1 - B.1) ^ 2 + (O.2 - B.2) ^ 2 :=
by sorry


end NUMINAMATH_CALUDE_determine_origin_l1263_126332


namespace NUMINAMATH_CALUDE_probability_sqrt_less_than_9_l1263_126392

/-- A two-digit whole number is an integer between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The count of two-digit whole numbers whose square root is less than 9. -/
def CountLessThan9 : ℕ := 71

/-- The total count of two-digit whole numbers. -/
def TotalTwoDigitNumbers : ℕ := 90

/-- The probability that the square root of a randomly selected two-digit whole number is less than 9. -/
theorem probability_sqrt_less_than_9 :
  (CountLessThan9 : ℚ) / TotalTwoDigitNumbers = 71 / 90 := by sorry

end NUMINAMATH_CALUDE_probability_sqrt_less_than_9_l1263_126392


namespace NUMINAMATH_CALUDE_arithmetic_seq_sum_l1263_126306

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence where S_9 = 72, a_2 + a_4 + a_9 = 24 -/
theorem arithmetic_seq_sum (seq : ArithmeticSequence) (h : seq.S 9 = 72) :
  seq.a 2 + seq.a 4 + seq.a 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_sum_l1263_126306


namespace NUMINAMATH_CALUDE_min_pencils_per_box_l1263_126364

/-- Represents a configuration of pencils in boxes -/
structure PencilConfiguration where
  num_boxes : Nat
  num_colors : Nat
  pencils_per_box : Nat

/-- Checks if a configuration satisfies the color requirement -/
def satisfies_color_requirement (config : PencilConfiguration) : Prop :=
  ∀ (subset : Finset (Fin config.num_boxes)), 
    subset.card = 4 → (subset.card * config.pencils_per_box ≥ config.num_colors)

/-- The main theorem stating the minimum number of pencils required -/
theorem min_pencils_per_box : 
  ∀ (config : PencilConfiguration),
    config.num_boxes = 6 ∧ 
    config.num_colors = 26 ∧ 
    satisfies_color_requirement config →
    config.pencils_per_box ≥ 13 := by
  sorry

end NUMINAMATH_CALUDE_min_pencils_per_box_l1263_126364


namespace NUMINAMATH_CALUDE_apple_sale_theorem_l1263_126396

/-- Calculates the total number of apples sold given the number of red apples and the ratio of red:green:yellow apples -/
def total_apples (red_apples : ℕ) (red_ratio green_ratio yellow_ratio : ℕ) : ℕ :=
  let total_ratio := red_ratio + green_ratio + yellow_ratio
  let apples_per_part := red_apples / red_ratio
  red_apples + (green_ratio * apples_per_part) + (yellow_ratio * apples_per_part)

/-- Theorem stating that given 32 red apples and a ratio of 8:3:5 for red:green:yellow apples, the total number of apples sold is 64 -/
theorem apple_sale_theorem : total_apples 32 8 3 5 = 64 := by
  sorry

end NUMINAMATH_CALUDE_apple_sale_theorem_l1263_126396


namespace NUMINAMATH_CALUDE_fraction_equality_l1263_126347

theorem fraction_equality (w x y z : ℝ) (hw : w ≠ 0) 
  (h : (x + 6*y - 3*z) / (-3*x + 4*w) = (-2*y + z) / (x - w) ∧ 
       (-2*y + z) / (x - w) = 2/3) : 
  x / w = 2/3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1263_126347


namespace NUMINAMATH_CALUDE_exponent_subtraction_l1263_126366

theorem exponent_subtraction : (-2)^3 - (-3)^2 = -17 := by
  sorry

end NUMINAMATH_CALUDE_exponent_subtraction_l1263_126366


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l1263_126330

/-- The total surface area of a hemisphere with base area 225π is 675π. -/
theorem hemisphere_surface_area : 
  ∀ r : ℝ, 
  r > 0 → 
  π * r^2 = 225 * π → 
  2 * π * r^2 + π * r^2 = 675 * π :=
by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l1263_126330


namespace NUMINAMATH_CALUDE_angle_of_inclination_sqrt_3_l1263_126386

/-- The angle of inclination (in radians) for a line with slope √3 is π/3 (60°) -/
theorem angle_of_inclination_sqrt_3 :
  let slope : ℝ := Real.sqrt 3
  let angle_of_inclination : ℝ := Real.arctan slope
  angle_of_inclination = π / 3 := by
sorry


end NUMINAMATH_CALUDE_angle_of_inclination_sqrt_3_l1263_126386


namespace NUMINAMATH_CALUDE_age_problem_l1263_126368

theorem age_problem (a b : ℚ) : 
  (a = 2 * (b - (a - b))) →  -- Condition 1
  (a + (a - b) + b + (a - b) = 130) →  -- Condition 2
  (a = 57 + 7/9 ∧ b = 43 + 1/3) := by
sorry

end NUMINAMATH_CALUDE_age_problem_l1263_126368


namespace NUMINAMATH_CALUDE_largest_unformable_amount_correct_l1263_126313

/-- Represents the coin denominations in Limonia -/
def coin_denominations (n : ℕ) : Finset ℕ := {3*n - 2, 6*n - 1, 6*n + 2, 6*n + 5}

/-- Predicate to check if an amount can be formed using given coin denominations -/
def is_formable (amount : ℕ) (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), amount = a*(3*n - 2) + b*(6*n - 1) + c*(6*n + 2) + d*(6*n + 5)

/-- The largest amount that cannot be formed using the coin denominations -/
def largest_unformable_amount (n : ℕ) : ℕ := 6*n^2 - 4*n - 3

/-- Main theorem: The largest amount that cannot be formed is 6n^2 - 4n - 3 -/
theorem largest_unformable_amount_correct (n : ℕ) :
  (∀ k > largest_unformable_amount n, is_formable k n) ∧
  ¬is_formable (largest_unformable_amount n) n :=
sorry

end NUMINAMATH_CALUDE_largest_unformable_amount_correct_l1263_126313


namespace NUMINAMATH_CALUDE_hire_year_proof_l1263_126360

/-- Rule of 70 provision: An employee can retire when their age plus years of employment total at least 70 -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year an employee was hired -/
def hire_year : ℕ := 1966

/-- The age at which the employee was hired -/
def hire_age : ℕ := 30

/-- The year the employee becomes eligible to retire -/
def retirement_eligibility_year : ℕ := 2006

/-- Theorem stating that an employee hired at age 30, who becomes eligible to retire under the rule of 70 provision in 2006, was hired in 1966 -/
theorem hire_year_proof :
  rule_of_70 (hire_age + (retirement_eligibility_year - hire_year)) (retirement_eligibility_year - hire_year) ∧
  hire_year = 1966 :=
sorry

end NUMINAMATH_CALUDE_hire_year_proof_l1263_126360


namespace NUMINAMATH_CALUDE_odd_function_has_zero_point_l1263_126336

theorem odd_function_has_zero_point (f : ℝ → ℝ) (h : ∀ x, f (-x) = -f x) :
  ∃ x, f x = 0 := by sorry

end NUMINAMATH_CALUDE_odd_function_has_zero_point_l1263_126336


namespace NUMINAMATH_CALUDE_eleven_billion_scientific_notation_l1263_126388

theorem eleven_billion_scientific_notation :
  (11 : ℝ) * (10 ^ 9 : ℝ) = (1.1 : ℝ) * (10 ^ 10 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_eleven_billion_scientific_notation_l1263_126388


namespace NUMINAMATH_CALUDE_unique_plane_through_line_and_point_l1263_126381

-- Define the 3D space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable [Fact (finrank ℝ V = 3)]

-- Define a line in 3D space
def Line (p q : V) : Set V :=
  {x | ∃ t : ℝ, x = p + t • (q - p)}

-- Define a plane in 3D space
def Plane (n : V) (c : ℝ) : Set V :=
  {x | inner n x = c}

-- State the theorem
theorem unique_plane_through_line_and_point 
  (l : Set V) (A : V) (p q : V) (h_line : l = Line p q) (h_not_on : A ∉ l) :
  ∃! P : Set V, ∃ n : V, ∃ c : ℝ, 
    P = Plane n c ∧ l ⊆ P ∧ A ∈ P :=
sorry

end NUMINAMATH_CALUDE_unique_plane_through_line_and_point_l1263_126381


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1263_126380

theorem inequality_solution_set (k : ℝ) :
  (∃ x : ℝ, |x - 2| - |x - 5| > k) → k < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1263_126380


namespace NUMINAMATH_CALUDE_inscribed_squares_product_l1263_126342

theorem inscribed_squares_product (a b : ℝ) : 
  (9 : ℝ).sqrt ^ 2 = 9 → 
  (16 : ℝ).sqrt ^ 2 = 16 → 
  a + b = (16 : ℝ).sqrt → 
  ((9 : ℝ).sqrt * Real.sqrt 2) ^ 2 = a ^ 2 + b ^ 2 → 
  a * b = -1 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_product_l1263_126342


namespace NUMINAMATH_CALUDE_cosine_equality_l1263_126301

theorem cosine_equality (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (942 * π / 180) → n = 138 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l1263_126301


namespace NUMINAMATH_CALUDE_coin_division_l1263_126393

theorem coin_division (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 6 ∨ m % 9 ≠ 7)) → 
  n % 8 = 6 → 
  n % 9 = 7 → 
  n % 11 = 8 := by
sorry

end NUMINAMATH_CALUDE_coin_division_l1263_126393
