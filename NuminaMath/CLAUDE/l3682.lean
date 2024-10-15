import Mathlib

namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3682_368231

theorem fractional_equation_solution :
  ∃ (x : ℝ), (3 / (x - 3) - 1 = 1 / (3 - x)) ∧ (x = 7) :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3682_368231


namespace NUMINAMATH_CALUDE_min_value_of_h_neg_infinity_to_zero_l3682_368202

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function h(x) defined in terms of f(x) and g(x) -/
def h (f g : ℝ → ℝ) (a b : ℝ) (x : ℝ) : ℝ := a * f x ^ 3 - b * g x - 2

theorem min_value_of_h_neg_infinity_to_zero 
  (f g : ℝ → ℝ) (a b : ℝ) 
  (hf : IsOdd f) (hg : IsOdd g)
  (hmax : ∃ x > 0, ∀ y > 0, h f g a b y ≤ h f g a b x ∧ h f g a b x = 5) :
  ∃ x < 0, ∀ y < 0, h f g a b y ≥ h f g a b x ∧ h f g a b x = -9 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_h_neg_infinity_to_zero_l3682_368202


namespace NUMINAMATH_CALUDE_square_cube_root_product_l3682_368253

theorem square_cube_root_product (a b : ℝ) 
  (ha : a^2 = 16/25) (hb : b^3 = 125/8) : 
  Real.sqrt (a * b) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_cube_root_product_l3682_368253


namespace NUMINAMATH_CALUDE_intersection_point_l3682_368228

-- Define the two linear functions
def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := -2*x + 6

-- State the theorem
theorem intersection_point :
  ∃! p : ℝ × ℝ, f p.1 = p.2 ∧ g p.1 = p.2 ∧ p = (1, 4) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3682_368228


namespace NUMINAMATH_CALUDE_red_face_probability_l3682_368210

/-- The volume of the original cube in cubic centimeters -/
def original_volume : ℝ := 27

/-- The number of small cubes the original cube is sawn into -/
def num_small_cubes : ℕ := 27

/-- The volume of each small cube in cubic centimeters -/
def small_cube_volume : ℝ := 1

/-- The number of small cubes with at least one red face -/
def num_red_cubes : ℕ := 26

/-- The probability of selecting a cube with at least one red face -/
def prob_red_face : ℚ := 26 / 27

theorem red_face_probability :
  original_volume = num_small_cubes * small_cube_volume →
  (num_red_cubes : ℚ) / num_small_cubes = prob_red_face := by
  sorry

end NUMINAMATH_CALUDE_red_face_probability_l3682_368210


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_sixth_powers_l3682_368224

theorem quadratic_roots_sum_of_sixth_powers (p q : ℂ) : 
  p^2 - 2*p*Real.sqrt 3 + 2 = 0 →
  q^2 - 2*q*Real.sqrt 3 + 2 = 0 →
  p^6 + q^6 = 3120 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_sixth_powers_l3682_368224


namespace NUMINAMATH_CALUDE_range_of_a_for_always_positive_quadratic_l3682_368234

theorem range_of_a_for_always_positive_quadratic :
  ∀ (a : ℝ), (∀ (x : ℝ), a * x^2 - 3 * a * x + 9 > 0) ↔ (0 ≤ a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_always_positive_quadratic_l3682_368234


namespace NUMINAMATH_CALUDE_product_range_l3682_368249

theorem product_range (a b : ℝ) (g : ℝ → ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : g = fun x => 2^x) (h₄ : g a * g b = 2) : 
  0 < a * b ∧ a * b ≤ 1/4 := by
sorry

end NUMINAMATH_CALUDE_product_range_l3682_368249


namespace NUMINAMATH_CALUDE_expression_simplification_l3682_368221

theorem expression_simplification (x : ℝ) (h : x + 2 = Real.sqrt 2) :
  ((x^2 + 1) / x + 2) / ((x - 3) * (x + 1) / (x^2 - 3*x)) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3682_368221


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l3682_368265

theorem floor_equation_solutions : 
  (∃ (S : Finset ℕ), S.card = 9 ∧ 
    (∀ x : ℕ, x ∈ S ↔ ⌊(x : ℚ) / 5⌋ = ⌊(x : ℚ) / 7⌋)) :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l3682_368265


namespace NUMINAMATH_CALUDE_fraction_sum_equals_point_three_l3682_368282

theorem fraction_sum_equals_point_three :
  (2 : ℚ) / 20 + (4 : ℚ) / 40 + (9 : ℚ) / 90 = (3 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_point_three_l3682_368282


namespace NUMINAMATH_CALUDE_island_not_maya_l3682_368215

-- Define the possible states for an inhabitant
inductive InhabitantState
  | Knight
  | Knave

-- Define the island name
structure IslandName where
  name : String

-- Define the statements made by the inhabitants
def statement_A (state_A state_B : InhabitantState) (island : IslandName) : Prop :=
  (state_A = InhabitantState.Knave ∨ state_B = InhabitantState.Knave) ∧ island.name = "Maya"

def statement_B (state_A state_B : InhabitantState) (island : IslandName) : Prop :=
  statement_A state_A state_B island

-- Define the truthfulness of statements based on the inhabitant's state
def is_truthful (state : InhabitantState) (statement : Prop) : Prop :=
  (state = InhabitantState.Knight ∧ statement) ∨ (state = InhabitantState.Knave ∧ ¬statement)

-- Theorem statement
theorem island_not_maya (state_A state_B : InhabitantState) (island : IslandName) :
  (is_truthful state_A (statement_A state_A state_B island) ∧
   is_truthful state_B (statement_B state_A state_B island)) →
  island.name ≠ "Maya" :=
by sorry

end NUMINAMATH_CALUDE_island_not_maya_l3682_368215


namespace NUMINAMATH_CALUDE_smallest_b_value_l3682_368241

theorem smallest_b_value (a b : ℕ+) (h1 : a - b = 8) 
  (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  ∀ c : ℕ+, c < b → ¬(∃ d : ℕ+, d - c = 8 ∧ 
    Nat.gcd ((d^3 + c^3) / (d + c)) (d * c) = 16) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l3682_368241


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l3682_368274

def g (x : ℝ) : ℝ := 16 * x^4 - 20 * x^2 + 5

theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt 5 / 2 ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l3682_368274


namespace NUMINAMATH_CALUDE_computer_table_markup_l3682_368252

/-- The percentage markup on a product's cost price, given the selling price and cost price. -/
def percentageMarkup (sellingPrice costPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

/-- Proof that the percentage markup on a computer table is 30% -/
theorem computer_table_markup :
  percentageMarkup 8450 6500 = 30 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_markup_l3682_368252


namespace NUMINAMATH_CALUDE_ted_speed_l3682_368269

theorem ted_speed (frank_speed : ℝ) (h1 : frank_speed > 0) : 
  let ted_speed := (2 / 3) * frank_speed
  2 * frank_speed = 2 * ted_speed + 8 →
  ted_speed = 8 := by
sorry

end NUMINAMATH_CALUDE_ted_speed_l3682_368269


namespace NUMINAMATH_CALUDE_infinitely_many_winning_starts_l3682_368209

theorem infinitely_many_winning_starts : 
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧ 
    ¬∃ k : ℕ, n = k^2 ∧
    ¬∃ k : ℕ, n + (n + 1) = k^2 ∧
    ∃ k : ℕ, (n + (n + 1)) + (n + 2) = k^2 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_winning_starts_l3682_368209


namespace NUMINAMATH_CALUDE_parabola_y_intercept_l3682_368250

/-- A parabola passing through two given points has a specific y-intercept -/
theorem parabola_y_intercept (b c : ℝ) : 
  ((-1 : ℝ)^2 + b*(-1) + c = -11) → 
  ((3 : ℝ)^2 + b*3 + c = 17) → 
  c = -7 := by
sorry

end NUMINAMATH_CALUDE_parabola_y_intercept_l3682_368250


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l3682_368200

/-- An ellipse with given foci and passing through specific points has the standard equation x²/8 + y²/4 = 1 -/
theorem ellipse_standard_equation (f1 f2 p1 p2 p3 : ℝ × ℝ) : 
  f1 = (0, -2) →
  f2 = (0, 2) →
  p1 = (-3/2, 5/2) →
  p2 = (2, -Real.sqrt 2) →
  p3 = (-1, Real.sqrt 14 / 2) →
  ∃ (ellipse : ℝ × ℝ → Prop),
    (∀ (x y : ℝ), ellipse (x, y) ↔ x^2/8 + y^2/4 = 1) ∧
    (ellipse f1 ∧ ellipse f2 ∧ ellipse p1 ∧ ellipse p2 ∧ ellipse p3) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l3682_368200


namespace NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l3682_368213

/-- Represents a line in a 2D Cartesian coordinate system -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Determines if a point (x, y) is in the fourth quadrant -/
def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Determines if a line passes through the fourth quadrant -/
def passes_through_fourth_quadrant (l : Line) : Prop :=
  ∃ x y : ℝ, y = l.slope * x + l.y_intercept ∧ is_in_fourth_quadrant x y

/-- The main theorem: the line y = 2x + 1 does not pass through the fourth quadrant -/
theorem line_not_in_fourth_quadrant :
  ¬ passes_through_fourth_quadrant (Line.mk 2 1) := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l3682_368213


namespace NUMINAMATH_CALUDE_marking_exists_l3682_368271

/-- Represents a 50x50 board with some cells occupied -/
def Board := Fin 50 → Fin 50 → Bool

/-- Represents a marking of free cells on the board -/
def Marking := Fin 50 → Fin 50 → Bool

/-- Check if a marking is valid (at most 99 cells marked) -/
def valid_marking (b : Board) (m : Marking) : Prop :=
  (Finset.sum Finset.univ (fun i => Finset.sum Finset.univ (fun j => if m i j then 1 else 0))) ≤ 99

/-- Check if the total number of marked and originally occupied cells in a row is even -/
def row_even (b : Board) (m : Marking) (i : Fin 50) : Prop :=
  Even (Finset.sum Finset.univ (fun j => if b i j || m i j then 1 else 0))

/-- Check if the total number of marked and originally occupied cells in a column is even -/
def col_even (b : Board) (m : Marking) (j : Fin 50) : Prop :=
  Even (Finset.sum Finset.univ (fun i => if b i j || m i j then 1 else 0))

/-- Main theorem: For any board configuration, there exists a valid marking that makes all rows and columns even -/
theorem marking_exists (b : Board) : ∃ m : Marking, 
  valid_marking b m ∧ 
  (∀ i : Fin 50, row_even b m i) ∧ 
  (∀ j : Fin 50, col_even b m j) := by
  sorry

end NUMINAMATH_CALUDE_marking_exists_l3682_368271


namespace NUMINAMATH_CALUDE_expression_factorization_l3682_368287

theorem expression_factorization (b : ℝ) : 
  (10 * b^4 - 27 * b^3 + 18 * b^2) - (-6 * b^4 + 4 * b^3 - 3 * b^2) = 
  b^2 * (16 * b - 7) * (b - 3) := by
sorry

end NUMINAMATH_CALUDE_expression_factorization_l3682_368287


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3682_368279

def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

def nth_term (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem arithmetic_sequence_problem (x y : ℝ) (m : ℕ) 
  (h1 : is_arithmetic_sequence (Real.log (x^2 * y^5)) (Real.log (x^4 * y^9)) (Real.log (x^7 * y^12)))
  (h2 : nth_term (Real.log (x^2 * y^5)) 
               ((Real.log (x^4 * y^9)) - (Real.log (x^2 * y^5))) 
               10 = Real.log (y^m)) :
  m = 55 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3682_368279


namespace NUMINAMATH_CALUDE_pool_filling_rounds_l3682_368255

/-- The number of buckets George can carry per round -/
def george_buckets : ℕ := 2

/-- The number of buckets Harry can carry per round -/
def harry_buckets : ℕ := 3

/-- The total number of buckets needed to fill the pool -/
def total_buckets : ℕ := 110

/-- The number of rounds needed to fill the pool -/
def rounds_to_fill : ℕ := total_buckets / (george_buckets + harry_buckets)

theorem pool_filling_rounds :
  rounds_to_fill = 22 := by sorry

end NUMINAMATH_CALUDE_pool_filling_rounds_l3682_368255


namespace NUMINAMATH_CALUDE_soccer_league_games_l3682_368230

/-- The number of games played in a soccer league -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a soccer league with 14 teams, where each team plays every other team once,
    the total number of games played is 91. -/
theorem soccer_league_games :
  num_games 14 = 91 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l3682_368230


namespace NUMINAMATH_CALUDE_value_of_S_6_l3682_368288

theorem value_of_S_6 (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12196 := by
  sorry

end NUMINAMATH_CALUDE_value_of_S_6_l3682_368288


namespace NUMINAMATH_CALUDE_max_remainder_division_by_nine_l3682_368281

theorem max_remainder_division_by_nine (n : ℕ) : 
  n / 9 = 6 → n % 9 ≤ 8 ∧ ∃ m : ℕ, m / 9 = 6 ∧ m % 9 = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_remainder_division_by_nine_l3682_368281


namespace NUMINAMATH_CALUDE_tan_three_properties_l3682_368290

theorem tan_three_properties (θ : Real) (h : Real.tan θ = 3) :
  (Real.cos θ / (Real.sin θ + 2 * Real.cos θ) = 1/5) ∧
  (Real.tan (θ - 5 * Real.pi / 4) = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_tan_three_properties_l3682_368290


namespace NUMINAMATH_CALUDE_nina_travel_distance_l3682_368267

theorem nina_travel_distance (x : ℕ) : 
  (12 * x + 12 * (2 * x) = 14400) → x = 400 := by
  sorry

end NUMINAMATH_CALUDE_nina_travel_distance_l3682_368267


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3682_368201

theorem quadratic_inequality_solution_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2 * m * x - 8 ≥ 0) ↔ m ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3682_368201


namespace NUMINAMATH_CALUDE_walking_time_difference_l3682_368289

/-- Proof of the walking time difference between Cara and Don --/
theorem walking_time_difference 
  (total_distance : ℝ) 
  (cara_speed : ℝ) 
  (don_speed : ℝ) 
  (cara_distance : ℝ) 
  (h1 : total_distance = 45) 
  (h2 : cara_speed = 6) 
  (h3 : don_speed = 5) 
  (h4 : cara_distance = 30) : 
  (cara_distance / cara_speed) - ((total_distance - cara_distance) / don_speed) = 2 := by
  sorry

end NUMINAMATH_CALUDE_walking_time_difference_l3682_368289


namespace NUMINAMATH_CALUDE_no_intersection_condition_l3682_368244

theorem no_intersection_condition (k : ℝ) : 
  -1 ≤ k ∧ k ≤ 1 → 
  (∀ x : ℝ, x = k * π / 2 → ¬∃ y : ℝ, y = Real.tan (2 * x + π / 4)) ↔ 
  (k = 1 / 4 ∨ k = -3 / 4) := by
sorry

end NUMINAMATH_CALUDE_no_intersection_condition_l3682_368244


namespace NUMINAMATH_CALUDE_task_completion_time_l3682_368223

/-- 
Given:
- Person A can complete a task in time a
- Person A and Person B together can complete the task in time c
- The rate of work is the reciprocal of the time taken

Prove:
- Person B can complete the task alone in time b, where 1/a + 1/b = 1/c
-/
theorem task_completion_time (a c : ℝ) (ha : a > 0) (hc : c > 0) (hac : c < a) :
  ∃ b : ℝ, b > 0 ∧ 1/a + 1/b = 1/c := by sorry

end NUMINAMATH_CALUDE_task_completion_time_l3682_368223


namespace NUMINAMATH_CALUDE_xyz_sum_l3682_368260

theorem xyz_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 32) (hxz : x * z = 64) (hyz : y * z = 96) :
  x + y + z = 44 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l3682_368260


namespace NUMINAMATH_CALUDE_chord_length_sum_l3682_368251

/-- Representation of a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Check if a circle is internally tangent to another -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c2.radius - c1.radius)^2

/-- Check if three points are collinear -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- Main theorem -/
theorem chord_length_sum (C1 C2 C3 : Circle) (m n p : ℕ) : 
  C1.radius = 4 →
  C2.radius = 10 →
  are_externally_tangent C1 C2 →
  is_internally_tangent C1 C3 →
  is_internally_tangent C2 C3 →
  are_collinear C1.center C2.center C3.center →
  (m.gcd p = 1) →
  (∀ q : ℕ, Prime q → n % (q^2) ≠ 0) →
  (∃ (chord_length : ℝ), chord_length = m * Real.sqrt n / p ∧ 
    chord_length^2 = 4 * (C3.radius^2 - ((C3.radius - C1.radius) * (C3.radius - C2.radius) / (C1.radius + C2.radius))^2)) →
  m + n + p = 405 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_sum_l3682_368251


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_of_cubes_l3682_368295

theorem consecutive_even_integers_sum_of_cubes (x y z : ℤ) : 
  (∃ n : ℤ, x = 2*n ∧ y = 2*n + 2 ∧ z = 2*n + 4) →  -- consecutive even integers
  x^2 + y^2 + z^2 = 2960 →                         -- sum of squares is 2960
  x^3 + y^3 + z^3 = 90117 :=                       -- sum of cubes is 90117
by sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_of_cubes_l3682_368295


namespace NUMINAMATH_CALUDE_author_writing_speed_l3682_368218

/-- Calculates the average words written per hour given total words, total hours, and break hours. -/
def average_words_per_hour (total_words : ℕ) (total_hours : ℕ) (break_hours : ℕ) : ℚ :=
  total_words / (total_hours - break_hours)

/-- Theorem stating that given the specific conditions, the average words per hour is 550. -/
theorem author_writing_speed :
  average_words_per_hour 55000 120 20 = 550 := by
  sorry

end NUMINAMATH_CALUDE_author_writing_speed_l3682_368218


namespace NUMINAMATH_CALUDE_mailbox_distance_l3682_368259

/-- Represents the problem of finding the distance to a mailbox --/
def MailboxProblem (initial_speed : ℝ) (return_speed : ℝ) (time_away : ℝ) : Prop :=
  let initial_speed_mpm := initial_speed * 1000 / 60
  let return_speed_mpm := return_speed * 1000 / 60
  let distance_mother_in_law := initial_speed_mpm * time_away
  let total_distance := return_speed_mpm * time_away
  let distance_to_mailbox := (total_distance + distance_mother_in_law) / 2
  distance_to_mailbox = 200

/-- The theorem stating the solution to the mailbox problem --/
theorem mailbox_distance :
  MailboxProblem 3 5 3 := by
  sorry


end NUMINAMATH_CALUDE_mailbox_distance_l3682_368259


namespace NUMINAMATH_CALUDE_square_reciprocal_sum_implies_fourth_power_reciprocal_sum_l3682_368219

theorem square_reciprocal_sum_implies_fourth_power_reciprocal_sum
  (x : ℝ) (h : x^2 + 1/x^2 = 6) : x^4 + 1/x^4 = 34 := by
  sorry

end NUMINAMATH_CALUDE_square_reciprocal_sum_implies_fourth_power_reciprocal_sum_l3682_368219


namespace NUMINAMATH_CALUDE_sum_remainder_seven_l3682_368280

theorem sum_remainder_seven (n : ℤ) : (7 - n + (n + 3)) % 7 = 3 := by sorry

end NUMINAMATH_CALUDE_sum_remainder_seven_l3682_368280


namespace NUMINAMATH_CALUDE_eliminate_quadratic_term_l3682_368248

/-- The polynomial we're working with -/
def polynomial (x n : ℝ) : ℝ := 4*x^2 + 2*(7 + 3*x - 3*x^2) - n*x^2

/-- The coefficient of x^2 in the expanded polynomial -/
def quadratic_coefficient (n : ℝ) : ℝ := 4 - 6 - n

theorem eliminate_quadratic_term :
  ∃ (n : ℝ), ∀ (x : ℝ), polynomial x n = 6*x + 14 ∧ n = -2 :=
sorry

end NUMINAMATH_CALUDE_eliminate_quadratic_term_l3682_368248


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3682_368263

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 - x₁ - 2 = 0) ∧ (x₂^2 - x₂ - 2 = 0) ∧ (x₁ = 2) ∧ (x₂ = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3682_368263


namespace NUMINAMATH_CALUDE_optimal_move_is_six_l3682_368264

/-- Represents the state of a number in the game -/
inductive NumberState
| Unmarked
| Marked
| Blocked

/-- Represents the game state -/
structure GameState where
  numbers : Fin 17 → NumberState

/-- Checks if a move is valid according to the game rules -/
def isValidMove (state : GameState) (n : Fin 17) : Prop :=
  state.numbers n = NumberState.Unmarked ∧
  ∀ m : Fin 17, state.numbers m = NumberState.Marked →
    n.val ≠ 2 * m.val ∧ n.val ≠ m.val / 2

/-- Applies a move to the game state -/
def applyMove (state : GameState) (n : Fin 17) : GameState :=
  { numbers := λ m =>
      if m = n then NumberState.Marked
      else if m.val = 2 * n.val ∨ 2 * m.val = n.val then NumberState.Blocked
      else state.numbers m }

/-- Checks if the game is over (no more valid moves) -/
def isGameOver (state : GameState) : Prop :=
  ∀ n : Fin 17, ¬isValidMove state n

/-- Defines the initial game state after A marks 8 -/
def initialState : GameState :=
  { numbers := λ n => if n.val = 8 then NumberState.Marked else NumberState.Unmarked }

/-- Theorem: B's optimal move is to mark 6 -/
theorem optimal_move_is_six :
  ∃ (strategy : GameState → Fin 17),
    (∀ state : GameState, isValidMove state (strategy state)) ∧
    (∀ (state : GameState) (n : Fin 17),
      isValidMove state n →
      isGameOver (applyMove (applyMove state (strategy state)) n)) ∧
    strategy initialState = ⟨6, by norm_num⟩ := by
  sorry


end NUMINAMATH_CALUDE_optimal_move_is_six_l3682_368264


namespace NUMINAMATH_CALUDE_cookie_radius_l3682_368245

theorem cookie_radius (x y : ℝ) : 
  (x^2 + y^2 - 6.5 = x + 3*y) → 
  ∃ (h k r : ℝ), r = 3 ∧ (x - h)^2 + (y - k)^2 = r^2 := by
  sorry

end NUMINAMATH_CALUDE_cookie_radius_l3682_368245


namespace NUMINAMATH_CALUDE_new_students_average_age_l3682_368237

/-- Given a class where:
    - The original number of students is 8
    - The original average age is 40 years
    - 8 new students join
    - The new average age of the entire class is 36 years
    This theorem proves that the average age of the new students is 32 years. -/
theorem new_students_average_age
  (original_count : Nat)
  (original_avg : ℝ)
  (new_count : Nat)
  (new_total_avg : ℝ)
  (h1 : original_count = 8)
  (h2 : original_avg = 40)
  (h3 : new_count = 8)
  (h4 : new_total_avg = 36) :
  (((original_count + new_count) * new_total_avg) - (original_count * original_avg)) / new_count = 32 := by
  sorry


end NUMINAMATH_CALUDE_new_students_average_age_l3682_368237


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3682_368242

/-- Proves that (7 + 14i) / (3 - 4i) = 77/25 + 70/25 * i -/
theorem complex_fraction_simplification :
  (7 + 14 * Complex.I) / (3 - 4 * Complex.I) = 77/25 + 70/25 * Complex.I :=
by sorry


end NUMINAMATH_CALUDE_complex_fraction_simplification_l3682_368242


namespace NUMINAMATH_CALUDE_truck_rental_charge_per_mile_l3682_368286

/-- Given a truck rental scenario, calculate the charge per mile. -/
theorem truck_rental_charge_per_mile
  (rental_fee : ℚ)
  (total_paid : ℚ)
  (miles_driven : ℕ)
  (h1 : rental_fee = 2099 / 100)
  (h2 : total_paid = 9574 / 100)
  (h3 : miles_driven = 299)
  : (total_paid - rental_fee) / miles_driven = 1 / 4 := by
  sorry

#eval (9574 / 100 : ℚ) - (2099 / 100 : ℚ)
#eval ((9574 / 100 : ℚ) - (2099 / 100 : ℚ)) / 299

end NUMINAMATH_CALUDE_truck_rental_charge_per_mile_l3682_368286


namespace NUMINAMATH_CALUDE_cubic_real_root_l3682_368227

theorem cubic_real_root (a b : ℝ) :
  (∃ x : ℂ, a * x^3 + 4 * x^2 + b * x - 35 = 0 ∧ x = -1 - 2*I) →
  (∃ x : ℝ, a * x^3 + 4 * x^2 + b * x - 35 = 0 ∧ x = 21/5) :=
by sorry

end NUMINAMATH_CALUDE_cubic_real_root_l3682_368227


namespace NUMINAMATH_CALUDE_fourth_term_of_solution_sequence_l3682_368217

def is_solution (x : ℤ) : Prop := x^2 - 2*x - 3 < 0

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fourth_term_of_solution_sequence :
  ∃ a : ℕ → ℤ,
    (∀ n : ℕ, is_solution (a n)) ∧
    arithmetic_sequence a ∧
    (a 4 = 3 ∨ a 4 = -1) := by sorry

end NUMINAMATH_CALUDE_fourth_term_of_solution_sequence_l3682_368217


namespace NUMINAMATH_CALUDE_odd_sum_floor_condition_l3682_368272

theorem odd_sum_floor_condition (p a b : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) 
  (ha : 0 < a ∧ a < p) (hb : 0 < b ∧ b < p) :
  (a + b = p) ↔ 
  (∀ n : ℕ, 0 < n → n < p → 
    ∃ k : ℕ, k % 2 = 1 ∧ 
      (⌊(2 * a * n : ℚ) / p⌋ + ⌊(2 * b * n : ℚ) / p⌋ : ℤ) = k) :=
by sorry

end NUMINAMATH_CALUDE_odd_sum_floor_condition_l3682_368272


namespace NUMINAMATH_CALUDE_six_playing_cards_distribution_l3682_368238

/-- Given a deck of cards with playing cards and instruction cards,
    distributed as evenly as possible among a group of people,
    calculate the number of people who end up with exactly 6 playing cards. -/
def people_with_six_playing_cards (total_cards : ℕ) (playing_cards : ℕ) (instruction_cards : ℕ) (num_people : ℕ) : ℕ :=
  let cards_per_person := total_cards / num_people
  let extra_cards := total_cards % num_people
  let playing_cards_distribution := playing_cards / num_people
  let extra_playing_cards := playing_cards % num_people
  min extra_playing_cards (num_people - instruction_cards)

theorem six_playing_cards_distribution :
  people_with_six_playing_cards 60 52 8 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_six_playing_cards_distribution_l3682_368238


namespace NUMINAMATH_CALUDE_fraction_of_three_fourths_half_5060_l3682_368220

theorem fraction_of_three_fourths_half_5060 : 
  let total := (3/4 : ℚ) * (1/2 : ℚ) * 5060
  759.0000000000001 / total = 0.4 := by sorry

end NUMINAMATH_CALUDE_fraction_of_three_fourths_half_5060_l3682_368220


namespace NUMINAMATH_CALUDE_meenas_bottle_caps_l3682_368203

theorem meenas_bottle_caps (initial : ℕ) : 
  (initial : ℚ) * (1 + 0.4) * (1 - 0.2) = initial + 21 → initial = 175 :=
by
  sorry

end NUMINAMATH_CALUDE_meenas_bottle_caps_l3682_368203


namespace NUMINAMATH_CALUDE_coffee_calculation_l3682_368294

/-- Calculates the total tablespoons of coffee needed for guests with different preferences -/
def total_coffee_tablespoons (total_guests : ℕ) : ℕ :=
  let weak_guests := total_guests / 3
  let medium_guests := total_guests / 3
  let strong_guests := total_guests - (weak_guests + medium_guests)
  let weak_cups := weak_guests * 2
  let medium_cups := medium_guests * 3
  let strong_cups := strong_guests * 1
  let weak_tablespoons := weak_cups * 1
  let medium_tablespoons := (medium_cups * 3) / 2
  let strong_tablespoons := strong_cups * 2
  weak_tablespoons + medium_tablespoons + strong_tablespoons

theorem coffee_calculation :
  total_coffee_tablespoons 18 = 51 := by
  sorry

end NUMINAMATH_CALUDE_coffee_calculation_l3682_368294


namespace NUMINAMATH_CALUDE_correct_proposition_l3682_368261

-- Define propositions p and q
variable (p q : Prop)

-- Define the conditions
axiom p_true : p
axiom q_false : ¬q

-- Theorem to prove
theorem correct_proposition : (¬p) ∨ (¬q) :=
sorry

end NUMINAMATH_CALUDE_correct_proposition_l3682_368261


namespace NUMINAMATH_CALUDE_road_area_in_square_park_l3682_368268

/-- 
Given a square park with a road inside, this theorem proves that
if the road is 3 meters wide and the perimeter along its outer edge is 600 meters,
then the area occupied by the road is 1836 square meters.
-/
theorem road_area_in_square_park (park_side : ℝ) (road_width : ℝ) (outer_perimeter : ℝ) 
  (h1 : road_width = 3)
  (h2 : outer_perimeter = 600)
  (h3 : 4 * (park_side - 2 * road_width) = outer_perimeter) :
  park_side^2 - (park_side - 2 * road_width)^2 = 1836 := by
  sorry

end NUMINAMATH_CALUDE_road_area_in_square_park_l3682_368268


namespace NUMINAMATH_CALUDE_certain_person_age_l3682_368258

def sandy_age : ℕ := 34
def person_age : ℕ := 10

theorem certain_person_age :
  (sandy_age * 10 = 340) →
  ((sandy_age + 2) = 3 * (person_age + 2)) →
  person_age = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_certain_person_age_l3682_368258


namespace NUMINAMATH_CALUDE_managers_in_sample_l3682_368270

structure StaffUnit where
  total : ℕ
  managers : ℕ
  sample_size : ℕ

def stratified_sample_size (unit : StaffUnit) (stratum_size : ℕ) : ℕ :=
  (stratum_size * unit.sample_size) / unit.total

theorem managers_in_sample (unit : StaffUnit) 
    (h1 : unit.total = 160)
    (h2 : unit.managers = 32)
    (h3 : unit.sample_size = 20) :
  stratified_sample_size unit unit.managers = 4 := by
  sorry

end NUMINAMATH_CALUDE_managers_in_sample_l3682_368270


namespace NUMINAMATH_CALUDE_redistribution_contribution_l3682_368226

theorem redistribution_contribution (earnings : Fin 5 → ℕ) 
  (h1 : earnings 0 = 18)
  (h2 : earnings 1 = 23)
  (h3 : earnings 2 = 30)
  (h4 : earnings 3 = 35)
  (h5 : earnings 4 = 50)
  (min_amount : ℕ := 30)
  : (earnings 4 - min_amount : ℕ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_redistribution_contribution_l3682_368226


namespace NUMINAMATH_CALUDE_average_steps_needed_l3682_368233

def goal : ℕ := 10000
def days : ℕ := 9
def remaining_days : ℕ := 3

def steps_walked : List ℕ := [10200, 10400, 9400, 9100, 8300, 9200, 8900, 9500]

def total_goal : ℕ := goal * days

def steps_walked_so_far : ℕ := steps_walked.sum

def remaining_steps : ℕ := total_goal - steps_walked_so_far

theorem average_steps_needed (h : steps_walked.length = days - remaining_days) :
  remaining_steps / remaining_days = 5000 := by
  sorry

end NUMINAMATH_CALUDE_average_steps_needed_l3682_368233


namespace NUMINAMATH_CALUDE_largest_rational_less_than_quarter_rank_3_l3682_368206

-- Define the rank of a rational number
def rank (q : ℚ) : ℕ :=
  -- The definition of rank is given in the problem statement
  sorry

-- Define the property of being the largest rational less than 1/4 with rank 3
def is_largest_less_than_quarter_rank_3 (q : ℚ) : Prop :=
  q < 1/4 ∧ rank q = 3 ∧ ∀ r, r < 1/4 ∧ rank r = 3 → r ≤ q

-- State the theorem
theorem largest_rational_less_than_quarter_rank_3 :
  ∃ q : ℚ, is_largest_less_than_quarter_rank_3 q ∧ q = 1/5 + 1/21 + 1/421 :=
sorry

end NUMINAMATH_CALUDE_largest_rational_less_than_quarter_rank_3_l3682_368206


namespace NUMINAMATH_CALUDE_reflect_across_y_axis_l3682_368225

/-- A point in a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The reflection of a point across the y-axis. -/
def reflectAcrossYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

/-- Theorem: The coordinates of a point P(x,y) with respect to the y-axis are (-x,y). -/
theorem reflect_across_y_axis (p : Point2D) :
  reflectAcrossYAxis p = { x := -p.x, y := p.y } := by
  sorry

#check reflect_across_y_axis

end NUMINAMATH_CALUDE_reflect_across_y_axis_l3682_368225


namespace NUMINAMATH_CALUDE_manu_win_probability_l3682_368243

def coin_flip_game (num_players : ℕ) (manu_position : ℕ) (manu_heads_needed : ℕ) : ℚ :=
  sorry

theorem manu_win_probability :
  coin_flip_game 4 4 2 = 1 / 30 := by sorry

end NUMINAMATH_CALUDE_manu_win_probability_l3682_368243


namespace NUMINAMATH_CALUDE_exterior_angle_measure_l3682_368214

/-- Given a regular polygon with sum of interior angles 1260°, 
    prove that each exterior angle measures 40° -/
theorem exterior_angle_measure (n : ℕ) : 
  (n - 2) * 180 = 1260 → 360 / n = 40 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_measure_l3682_368214


namespace NUMINAMATH_CALUDE_exists_divisor_in_range_l3682_368273

theorem exists_divisor_in_range : ∃ n : ℕ, 
  100 ≤ n ∧ n ≤ 1997 ∧ (n ∣ 2 * n + 2) ∧ n = 946 := by
  sorry

end NUMINAMATH_CALUDE_exists_divisor_in_range_l3682_368273


namespace NUMINAMATH_CALUDE_equation_solutions_l3682_368283

theorem equation_solutions :
  (∀ x : ℝ, 16 * x^2 = 49 ↔ x = 7/4 ∨ x = -7/4) ∧
  (∀ x : ℝ, (x - 2)^2 = 64 ↔ x = 10 ∨ x = -6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3682_368283


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l3682_368292

/-- A point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line defined by two points -/
structure Line :=
  (p1 : Point)
  (p2 : Point)

/-- The x-axis -/
def x_axis : Line :=
  { p1 := ⟨0, 0⟩, p2 := ⟨1, 0⟩ }

/-- Function to determine if a point lies on a given line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

/-- Function to determine if a point lies on the x-axis -/
def point_on_x_axis (p : Point) : Prop :=
  p.y = 0

/-- The main theorem -/
theorem line_intersection_x_axis :
  let l : Line := { p1 := ⟨7, 3⟩, p2 := ⟨3, 7⟩ }
  let intersection_point : Point := ⟨10, 0⟩
  point_on_line intersection_point l ∧ point_on_x_axis intersection_point :=
by
  sorry

end NUMINAMATH_CALUDE_line_intersection_x_axis_l3682_368292


namespace NUMINAMATH_CALUDE_complex_absolute_value_equation_l3682_368291

theorem complex_absolute_value_equation :
  ∃ (t : ℝ), t > 0 ∧ Complex.abs (-3 + t * Complex.I) = 3 * Real.sqrt 10 ∧ t = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_equation_l3682_368291


namespace NUMINAMATH_CALUDE_odd_increasing_function_property_l3682_368256

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is monotonically increasing if x < y implies f(x) < f(y) -/
def IsMonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem odd_increasing_function_property (f : ℝ → ℝ) 
    (h_odd : IsOdd f) (h_mono : IsMonoIncreasing f) :
    (∀ a b : ℝ, f a + f (b - 1) = 0) → 
    (∀ a b : ℝ, a + b = 1) :=
  sorry

end NUMINAMATH_CALUDE_odd_increasing_function_property_l3682_368256


namespace NUMINAMATH_CALUDE_fred_seashells_l3682_368277

def seashells_problem (initial_seashells given_away_seashells : ℕ) : Prop :=
  initial_seashells - given_away_seashells = 22

theorem fred_seashells : seashells_problem 47 25 := by
  sorry

end NUMINAMATH_CALUDE_fred_seashells_l3682_368277


namespace NUMINAMATH_CALUDE_gcd_of_75_and_100_l3682_368297

theorem gcd_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_75_and_100_l3682_368297


namespace NUMINAMATH_CALUDE_equality_of_ratios_implies_k_eighteen_l3682_368232

theorem equality_of_ratios_implies_k_eighteen 
  (x y z k : ℝ) 
  (h : (7 : ℝ) / (x + y) = k / (x + z) ∧ k / (x + z) = (11 : ℝ) / (z - y)) : 
  k = 18 := by
sorry

end NUMINAMATH_CALUDE_equality_of_ratios_implies_k_eighteen_l3682_368232


namespace NUMINAMATH_CALUDE_turtle_ratio_l3682_368229

/-- Prove that given the conditions, the ratio of turtles Kris has to Kristen has is 1:4 -/
theorem turtle_ratio : 
  ∀ (kris trey kristen : ℕ),
  trey = 5 * kris →
  kris + trey + kristen = 30 →
  kristen = 12 →
  kris.gcd kristen = 3 →
  (kris / 3 : ℚ) / (kristen / 3 : ℚ) = 1 / 4 := by
sorry


end NUMINAMATH_CALUDE_turtle_ratio_l3682_368229


namespace NUMINAMATH_CALUDE_johns_allowance_l3682_368275

theorem johns_allowance (A : ℝ) : A = 2.40 ↔ 
  ∃ (arcade_spent toy_store_spent candy_store_spent : ℝ),
    arcade_spent = (3/5) * A ∧
    toy_store_spent = (1/3) * (A - arcade_spent) ∧
    candy_store_spent = A - arcade_spent - toy_store_spent ∧
    candy_store_spent = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_l3682_368275


namespace NUMINAMATH_CALUDE_passing_train_speed_is_50_l3682_368257

/-- The speed of the passing train in km/h -/
def passing_train_speed : ℝ := 50

/-- The speed of the passenger's train in km/h -/
def passenger_train_speed : ℝ := 40

/-- The time taken for the passing train to pass completely in seconds -/
def passing_time : ℝ := 3

/-- The length of the passing train in meters -/
def passing_train_length : ℝ := 75

/-- Theorem stating that the speed of the passing train is 50 km/h -/
theorem passing_train_speed_is_50 :
  passing_train_speed = 50 :=
sorry

end NUMINAMATH_CALUDE_passing_train_speed_is_50_l3682_368257


namespace NUMINAMATH_CALUDE_product_digit_sum_l3682_368211

def a : ℕ := 70707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def b : ℕ := 60606060606060606060606060606060606060606060606060606060606060606060606060606060606060606060606060606

theorem product_digit_sum :
  (a * b % 10) + ((a * b / 10000) % 10) = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l3682_368211


namespace NUMINAMATH_CALUDE_union_A_complementB_equals_result_l3682_368240

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x | x^2 - 2*x < 0}

-- Define the complement of B
def complementB : Set ℝ := {x | ¬(x ∈ B)}

-- Define the result set
def result : Set ℝ := {x | x ≤ 1 ∨ 2 ≤ x}

-- Theorem statement
theorem union_A_complementB_equals_result : A ∪ complementB = result := by
  sorry

end NUMINAMATH_CALUDE_union_A_complementB_equals_result_l3682_368240


namespace NUMINAMATH_CALUDE_quadratic_roots_average_l3682_368293

theorem quadratic_roots_average (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 - 4 * a * x + b
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 →
  (x + y) / 2 = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_average_l3682_368293


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3682_368299

theorem algebraic_expression_value (a b : ℝ) (h : a - b = 2 * Real.sqrt 3) :
  ((a^2 + b^2) / (2 * a) - b) * (a / (a - b)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3682_368299


namespace NUMINAMATH_CALUDE_son_work_time_l3682_368284

/-- Given a man can do a piece of work in 5 days, and together with his son they can do it in 3 days,
    prove that the son can do the work alone in 7.5 days. -/
theorem son_work_time (man_time : ℝ) (combined_time : ℝ) (son_time : ℝ) 
    (h1 : man_time = 5)
    (h2 : combined_time = 3) :
    son_time = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_son_work_time_l3682_368284


namespace NUMINAMATH_CALUDE_focus_to_line_distance_l3682_368285

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line
def line (x y : ℝ) : Prop := Real.sqrt 3 * x - y = 0

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- State the theorem
theorem focus_to_line_distance :
  let (fx, fy) := focus
  ∃ d : ℝ, d = Real.sqrt 3 ∧
    d = |Real.sqrt 3 * fx - fy| / Real.sqrt (3 + 1) :=
sorry

end NUMINAMATH_CALUDE_focus_to_line_distance_l3682_368285


namespace NUMINAMATH_CALUDE_total_count_equals_115248_l3682_368278

/-- The number of digits that can be used (excluding 3, 6, and 9) -/
def available_digits : ℕ := 7

/-- The number of non-zero digits that can be used as the first digit -/
def first_digit_choices : ℕ := 6

/-- Calculates the number of n-digit numbers without 3, 6, or 9 -/
def count_numbers (n : ℕ) : ℕ :=
  first_digit_choices * available_digits^(n - 1)

/-- The total number of 5 and 6-digit numbers without 3, 6, or 9 -/
def total_count : ℕ := count_numbers 5 + count_numbers 6

theorem total_count_equals_115248 : total_count = 115248 := by
  sorry

end NUMINAMATH_CALUDE_total_count_equals_115248_l3682_368278


namespace NUMINAMATH_CALUDE_function_identity_l3682_368246

theorem function_identity (f : ℚ → ℚ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) : 
  ∀ x : ℚ, f x = x + 1 := by
sorry

end NUMINAMATH_CALUDE_function_identity_l3682_368246


namespace NUMINAMATH_CALUDE_work_completion_theorem_l3682_368254

/-- The number of men originally employed to finish the work in 11 days -/
def original_men : ℕ := 27

/-- The number of additional men who joined -/
def additional_men : ℕ := 10

/-- The original number of days to finish the work -/
def original_days : ℕ := 11

/-- The number of days saved after additional men joined -/
def days_saved : ℕ := 3

theorem work_completion_theorem :
  original_men + additional_men = 37 ∧
  original_men * original_days = (original_men + additional_men) * (original_days - days_saved) :=
sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l3682_368254


namespace NUMINAMATH_CALUDE_geometric_place_of_tangent_points_l3682_368236

/-- Given a circle with center O(0,0) and radius r in a right-angled coordinate system,
    the geometric place of points S(x,y) whose adjoint lines are tangents to the circle
    is defined by the equation 1/x^2 + 1/y^2 = 1/r^2 -/
theorem geometric_place_of_tangent_points (r : ℝ) (h : r > 0) :
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 →
    (∃ x₁ y₁ : ℝ, x₁^2 + y₁^2 = r^2 ∧ x₁ * x + y₁ * y = r^2) ↔
    1 / x^2 + 1 / y^2 = 1 / r^2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_place_of_tangent_points_l3682_368236


namespace NUMINAMATH_CALUDE_simplify_expression_l3682_368235

theorem simplify_expression : 
  (81 ^ (1/4) - Real.sqrt (17/2)) ^ 2 = 17.5 - 3 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3682_368235


namespace NUMINAMATH_CALUDE_abs_is_even_and_increasing_l3682_368266

def f (x : ℝ) := |x|

theorem abs_is_even_and_increasing :
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_abs_is_even_and_increasing_l3682_368266


namespace NUMINAMATH_CALUDE_four_noncoplanar_points_determine_four_planes_l3682_368208

-- Define a Point type
def Point : Type := ℝ × ℝ × ℝ

-- Define a Plane type
structure Plane where
  normal : ℝ × ℝ × ℝ
  d : ℝ

-- Define a function to check if points are coplanar
def are_coplanar (p1 p2 p3 p4 : Point) : Prop := sorry

-- Define a function to create a plane from three points
def plane_from_points (p1 p2 p3 : Point) : Plane := sorry

-- Define a function to count the number of unique planes
def count_unique_planes (planes : List Plane) : Nat := sorry

-- Theorem statement
theorem four_noncoplanar_points_determine_four_planes 
  (p1 p2 p3 p4 : Point) 
  (h : ¬ are_coplanar p1 p2 p3 p4) : 
  count_unique_planes [
    plane_from_points p1 p2 p3,
    plane_from_points p1 p2 p4,
    plane_from_points p1 p3 p4,
    plane_from_points p2 p3 p4
  ] = 4 := by sorry

end NUMINAMATH_CALUDE_four_noncoplanar_points_determine_four_planes_l3682_368208


namespace NUMINAMATH_CALUDE_min_quotient_is_20_5_l3682_368216

/-- Represents a three-digit number with digits a, b, and c -/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  a_nonzero : a > 0
  b_nonzero : b > 0
  c_nonzero : c > 0
  all_different : a ≠ b ∧ b ≠ c ∧ a ≠ c
  b_relation : b = a + 1
  c_relation : c = b + 1

/-- The quotient of the number divided by the sum of its digits -/
def quotient (n : ThreeDigitNumber) : ℚ :=
  (100 * n.a + 10 * n.b + n.c) / (n.a + n.b + n.c)

/-- The theorem stating that the minimum quotient is 20.5 -/
theorem min_quotient_is_20_5 :
  ∀ n : ThreeDigitNumber, quotient n ≥ 20.5 ∧ ∃ n : ThreeDigitNumber, quotient n = 20.5 :=
sorry

end NUMINAMATH_CALUDE_min_quotient_is_20_5_l3682_368216


namespace NUMINAMATH_CALUDE_problem_solution_l3682_368239

theorem problem_solution (m n : ℝ) : 
  (∃ k : ℝ, k^2 = 3*m + 1 ∧ (k = 2 ∨ k = -2)) →
  (∃ l : ℝ, l^3 = 5*n - 2 ∧ l = 2) →
  m = 1 ∧ n = 2 ∧ (∃ r : ℝ, r^2 = 4*m + 5/2*n ∧ (r = 3 ∨ r = -3)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3682_368239


namespace NUMINAMATH_CALUDE_books_per_student_l3682_368207

theorem books_per_student (total_books : ℕ) (students_day1 students_day2 students_day3 students_day4 : ℕ) : 
  total_books = 120 →
  students_day1 = 4 →
  students_day2 = 5 →
  students_day3 = 6 →
  students_day4 = 9 →
  total_books / (students_day1 + students_day2 + students_day3 + students_day4) = 5 := by
sorry

end NUMINAMATH_CALUDE_books_per_student_l3682_368207


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3682_368222

-- Part 1
def f (x : ℝ) : ℝ := |x - 1| - 2

theorem solution_set_part1 :
  {x : ℝ | f x + |2*x - 3| > 0} = {x : ℝ | x > 2 ∨ x < 2/3} := by sorry

-- Part 2
def g (a x : ℝ) : ℝ := |x - a| - 2

theorem range_of_a_part2 (a : ℝ) :
  (∃ x, g a x > |x - 3|) → a < 1 ∨ a > 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3682_368222


namespace NUMINAMATH_CALUDE_horner_v₁_value_l3682_368262

def f (x : ℝ) : ℝ := 3 * x^6 + 4 * x^5 + 5 * x^4 + 6 * x^3 + 7 * x^2 + 8 * x + 1

def horner_v₁ (a₆ a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((((a₆ * x + a₅) * x + a₄) * x + a₃) * x + a₂) * x + a₁

theorem horner_v₁_value :
  horner_v₁ 3 4 5 6 7 8 1 0.4 = 5.2 :=
sorry

end NUMINAMATH_CALUDE_horner_v₁_value_l3682_368262


namespace NUMINAMATH_CALUDE_mika_bought_26_stickers_l3682_368204

/-- Represents the number of stickers Mika has at different stages -/
structure StickerCount where
  initial : Nat
  birthday : Nat
  given_away : Nat
  used : Nat
  remaining : Nat

/-- Calculates the number of stickers Mika bought from the store -/
def stickers_bought (s : StickerCount) : Nat :=
  s.remaining + s.given_away + s.used - s.initial - s.birthday

/-- Theorem stating that Mika bought 26 stickers from the store -/
theorem mika_bought_26_stickers (s : StickerCount) 
  (h1 : s.initial = 20)
  (h2 : s.birthday = 20)
  (h3 : s.given_away = 6)
  (h4 : s.used = 58)
  (h5 : s.remaining = 2) :
  stickers_bought s = 26 := by
  sorry

#eval stickers_bought { initial := 20, birthday := 20, given_away := 6, used := 58, remaining := 2 }

end NUMINAMATH_CALUDE_mika_bought_26_stickers_l3682_368204


namespace NUMINAMATH_CALUDE_problem_statement_l3682_368276

theorem problem_statement (a b c x y z : ℝ) 
  (h1 : x^2 - y^2 - z^2 = 2*a*y*z)
  (h2 : -x^2 + y^2 - z^2 = 2*b*z*x)
  (h3 : -x^2 - y^2 + z^2 = 2*c*x*y)
  (h4 : x*y*z ≠ 0) :
  a^2 + b^2 + c^2 - 2*a*b*c = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3682_368276


namespace NUMINAMATH_CALUDE_shooting_competition_probability_l3682_368205

theorem shooting_competition_probability (p : ℝ) (n : ℕ) (k : ℕ) : 
  p = 0.4 → n = 3 → k = 2 →
  (Finset.sum (Finset.range (n + 1 - k)) (λ i => Nat.choose n (n - i) * p^(n - i) * (1 - p)^i)) = 0.352 := by
sorry

end NUMINAMATH_CALUDE_shooting_competition_probability_l3682_368205


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_two_l3682_368212

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_ratio_two (a : ℕ → ℝ) 
  (h : geometric_sequence a) 
  (h_ratio : ∀ n : ℕ, a (n + 1) = 2 * a n) :
  (2 * a 2 + a 3) / (2 * a 4 + a 5) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_two_l3682_368212


namespace NUMINAMATH_CALUDE_triangle_inequality_l3682_368247

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (-a^2 + b^2 + c^2) * (a^2 - b^2 + c^2) * (a^2 + b^2 - c^2) ≤ a^2 * b^2 * c^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3682_368247


namespace NUMINAMATH_CALUDE_annas_age_problem_l3682_368298

theorem annas_age_problem :
  ∃! x : ℕ+, 
    (∃ m : ℕ, (x : ℤ) - 4 = m^2) ∧ 
    (∃ n : ℕ, (x : ℤ) + 3 = n^3) ∧ 
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_annas_age_problem_l3682_368298


namespace NUMINAMATH_CALUDE_bottles_per_crate_l3682_368296

theorem bottles_per_crate 
  (total_bottles : ℕ) 
  (num_crates : ℕ) 
  (unpacked_bottles : ℕ) 
  (h1 : total_bottles = 130) 
  (h2 : num_crates = 10) 
  (h3 : unpacked_bottles = 10) 
  : (total_bottles - unpacked_bottles) / num_crates = 12 := by
  sorry

end NUMINAMATH_CALUDE_bottles_per_crate_l3682_368296
