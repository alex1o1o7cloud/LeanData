import Mathlib

namespace NUMINAMATH_CALUDE_nested_sqrt_eighteen_l1284_128477

theorem nested_sqrt_eighteen (y : ℝ) : y = Real.sqrt (18 + y) → y = (1 + Real.sqrt 73) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_eighteen_l1284_128477


namespace NUMINAMATH_CALUDE_simplify_expression_l1284_128493

theorem simplify_expression (a : ℝ) (ha : a > 0) :
  (15 / 8) * Real.sqrt (2 + 10 / 27) / Real.sqrt (25 / (12 * a^3)) = a * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1284_128493


namespace NUMINAMATH_CALUDE_probability_three_divisible_by_3_l1284_128409

/-- The probability of a single 12-sided die showing a number divisible by 3 -/
def p_divisible_by_3 : ℚ := 1 / 3

/-- The probability of a single 12-sided die not showing a number divisible by 3 -/
def p_not_divisible_by_3 : ℚ := 2 / 3

/-- The number of dice rolled -/
def total_dice : ℕ := 7

/-- The number of dice that should show a number divisible by 3 -/
def target_dice : ℕ := 3

/-- The theorem stating the probability of exactly three out of seven fair 12-sided dice 
    showing a number divisible by 3 -/
theorem probability_three_divisible_by_3 : 
  (Nat.choose total_dice target_dice : ℚ) * 
  p_divisible_by_3 ^ target_dice * 
  p_not_divisible_by_3 ^ (total_dice - target_dice) = 560 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_divisible_by_3_l1284_128409


namespace NUMINAMATH_CALUDE_solution_set_l1284_128467

theorem solution_set (x : ℝ) :
  x > 9 →
  Real.sqrt (x - 6 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 9)) - 3 →
  x ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_solution_set_l1284_128467


namespace NUMINAMATH_CALUDE_inscribed_triangle_property_l1284_128499

-- Define the triangle XYZ
def Triangle (X Y Z : ℝ × ℝ) : Prop :=
  let xy := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  let yz := Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2)
  let xz := Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2)
  xy = 26 ∧ yz = 28 ∧ xz = 27

-- Define the inscribed triangle GHI
def InscribedTriangle (X Y Z G H I : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ t₃ : ℝ,
    0 < t₁ ∧ t₁ < 1 ∧
    0 < t₂ ∧ t₂ < 1 ∧
    0 < t₃ ∧ t₃ < 1 ∧
    G = (t₁ * Y.1 + (1 - t₁) * Z.1, t₁ * Y.2 + (1 - t₁) * Z.2) ∧
    H = (t₂ * X.1 + (1 - t₂) * Z.1, t₂ * X.2 + (1 - t₂) * Z.2) ∧
    I = (t₃ * X.1 + (1 - t₃) * Y.1, t₃ * X.2 + (1 - t₃) * Y.2)

-- Define the equality of arcs
def ArcEqual (X Y Z G H I : ℝ × ℝ) : Prop :=
  let yi := Real.sqrt ((Y.1 - I.1)^2 + (Y.2 - I.2)^2)
  let gz := Real.sqrt ((G.1 - Z.1)^2 + (G.2 - Z.2)^2)
  let xi := Real.sqrt ((X.1 - I.1)^2 + (X.2 - I.2)^2)
  let hz := Real.sqrt ((H.1 - Z.1)^2 + (H.2 - Z.2)^2)
  let xh := Real.sqrt ((X.1 - H.1)^2 + (X.2 - H.2)^2)
  let gy := Real.sqrt ((G.1 - Y.1)^2 + (G.2 - Y.2)^2)
  yi = gz ∧ xi = hz ∧ xh = gy

theorem inscribed_triangle_property
  (X Y Z G H I : ℝ × ℝ)
  (h₁ : Triangle X Y Z)
  (h₂ : InscribedTriangle X Y Z G H I)
  (h₃ : ArcEqual X Y Z G H I) :
  let gy := Real.sqrt ((G.1 - Y.1)^2 + (G.2 - Y.2)^2)
  gy = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_property_l1284_128499


namespace NUMINAMATH_CALUDE_meet_time_l1284_128441

/-- Represents the scenario of Petya and Vasya's journey --/
structure Journey where
  distance : ℝ  -- Total distance between Petya and Vasya
  speed_dirt : ℝ  -- Speed on dirt road
  speed_paved : ℝ  -- Speed on paved road
  time_to_bridge : ℝ  -- Time for Petya to reach the bridge

/-- The conditions of the journey --/
def journey_conditions (j : Journey) : Prop :=
  j.speed_paved = 3 * j.speed_dirt ∧
  j.time_to_bridge = 1 ∧
  j.distance / 2 = j.speed_paved * j.time_to_bridge

/-- The theorem to be proved --/
theorem meet_time (j : Journey) (h : journey_conditions j) : 
  ∃ (t : ℝ), t = 2 ∧ t = j.time_to_bridge + (j.distance / 2 - j.speed_dirt * j.time_to_bridge) / (2 * j.speed_dirt) :=
sorry

end NUMINAMATH_CALUDE_meet_time_l1284_128441


namespace NUMINAMATH_CALUDE_banana_weights_l1284_128440

/-- A scale with a constant displacement --/
structure DisplacedScale where
  displacement : ℝ

/-- Measurements of banana bunches on a displaced scale --/
structure BananaMeasurements where
  small_bunch : ℝ
  large_bunch : ℝ
  combined_bunches : ℝ

/-- The actual weights of the banana bunches --/
def actual_weights (s : DisplacedScale) (m : BananaMeasurements) : Prop :=
  ∃ (small large : ℝ),
    small = m.small_bunch - s.displacement ∧
    large = m.large_bunch - s.displacement ∧
    small + large = m.combined_bunches - s.displacement ∧
    small = 1 ∧ large = 2

/-- Theorem stating that given the measurements, the actual weights are 1 kg and 2 kg --/
theorem banana_weights (s : DisplacedScale) (m : BananaMeasurements) 
  (h1 : m.small_bunch = 1.5)
  (h2 : m.large_bunch = 2.5)
  (h3 : m.combined_bunches = 3.5) :
  actual_weights s m :=
by sorry

end NUMINAMATH_CALUDE_banana_weights_l1284_128440


namespace NUMINAMATH_CALUDE_angle_c_measure_l1284_128458

theorem angle_c_measure (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_measure_l1284_128458


namespace NUMINAMATH_CALUDE_inequality_range_l1284_128455

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ m^2 - 3*m) → 
  (m ≥ 4 ∨ m ≤ -1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1284_128455


namespace NUMINAMATH_CALUDE_wood_per_chair_l1284_128414

def total_wood : ℕ := 672
def wood_per_table : ℕ := 12
def num_tables : ℕ := 24
def num_chairs : ℕ := 48

theorem wood_per_chair :
  (total_wood - num_tables * wood_per_table) / num_chairs = 8 := by
  sorry

end NUMINAMATH_CALUDE_wood_per_chair_l1284_128414


namespace NUMINAMATH_CALUDE_right_triangle_sides_l1284_128435

/-- A right triangle with perimeter 60 and height to hypotenuse 12 has sides 15, 20, and 35 -/
theorem right_triangle_sides (a b c : ℝ) (h : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 →
  a + b + c = 60 →
  a^2 + b^2 = c^2 →
  a * b = 12 * c →
  h = 12 →
  (a = 15 ∧ b = 20 ∧ c = 35) ∨ (a = 20 ∧ b = 15 ∧ c = 35) :=
by sorry


end NUMINAMATH_CALUDE_right_triangle_sides_l1284_128435


namespace NUMINAMATH_CALUDE_union_of_A_and_B_intersection_empty_iff_l1284_128437

def A (m : ℝ) : Set ℝ := {x | 2*m - 1 < x ∧ x < m}
def B : Set ℝ := {x | -4 ≤ x ∧ x ≤ 5}

theorem union_of_A_and_B (m : ℝ) :
  m = -3 → A m ∪ B = {x | -7 < x ∧ x ≤ 5} := by sorry

theorem intersection_empty_iff (m : ℝ) :
  A m ∩ B = ∅ ↔ m ≤ -4 ∨ 1 ≤ m := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_intersection_empty_iff_l1284_128437


namespace NUMINAMATH_CALUDE_sally_quarters_problem_l1284_128490

theorem sally_quarters_problem (initial_quarters : ℕ) 
  (first_purchase : ℕ) (second_purchase : ℕ) :
  initial_quarters = 760 →
  first_purchase = 418 →
  second_purchase = 215 →
  initial_quarters - first_purchase - second_purchase = 127 :=
by sorry

end NUMINAMATH_CALUDE_sally_quarters_problem_l1284_128490


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1284_128476

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating that f(3^x) ≤ f(2^x) for all real x, 
    given that f is monotonically increasing on (-∞, 1] -/
theorem quadratic_inequality (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 1 → f a b c x ≤ f a b c y) →
  ∀ x : ℝ, f a b c (3^x) ≤ f a b c (2^x) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1284_128476


namespace NUMINAMATH_CALUDE_point_distance_from_y_axis_l1284_128445

theorem point_distance_from_y_axis (a : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (a - 3, 2 * a) ∧ |a - 3| = 2) → 
  (a = 5 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_point_distance_from_y_axis_l1284_128445


namespace NUMINAMATH_CALUDE_total_chairs_bought_l1284_128449

def living_room_chairs : ℕ := 3
def kitchen_chairs : ℕ := 6

theorem total_chairs_bought : living_room_chairs + kitchen_chairs = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_chairs_bought_l1284_128449


namespace NUMINAMATH_CALUDE_no_goal_scored_l1284_128404

def football_play (play1 play2 play3 play4 : ℝ) : Prop :=
  play1 = -5 ∧ 
  play2 = 13 ∧ 
  play3 = -(play1^2) ∧ 
  play4 = -play3 / 2

def total_progress (play1 play2 play3 play4 : ℝ) : ℝ :=
  play1 + play2 + play3 + play4

def score_goal (progress : ℝ) : Prop :=
  progress ≥ 30

theorem no_goal_scored (play1 play2 play3 play4 : ℝ) :
  football_play play1 play2 play3 play4 →
  ¬(score_goal (total_progress play1 play2 play3 play4)) :=
by sorry

end NUMINAMATH_CALUDE_no_goal_scored_l1284_128404


namespace NUMINAMATH_CALUDE_absolute_difference_inequality_l1284_128431

theorem absolute_difference_inequality (x y : ℝ) 
  (hx : |x| < 1) (hy : |y| < 1) : 
  |x - y| < |1 - x*y| := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_inequality_l1284_128431


namespace NUMINAMATH_CALUDE_inequality_range_l1284_128401

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x - 3| - |x + 1| ≤ a^2 - 3*a) ↔ (a ≤ -1 ∨ a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l1284_128401


namespace NUMINAMATH_CALUDE_fraction_not_simplifiable_l1284_128468

theorem fraction_not_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_not_simplifiable_l1284_128468


namespace NUMINAMATH_CALUDE_chairs_per_row_l1284_128471

theorem chairs_per_row (total_chairs : ℕ) (num_rows : ℕ) (h1 : total_chairs = 432) (h2 : num_rows = 27) :
  total_chairs / num_rows = 16 := by
  sorry

end NUMINAMATH_CALUDE_chairs_per_row_l1284_128471


namespace NUMINAMATH_CALUDE_square_perimeter_from_rectangle_perimeter_l1284_128443

/-- Given a square divided into six congruent rectangles, if each rectangle has a perimeter of 30 inches, then the perimeter of the square is 360/7 inches. -/
theorem square_perimeter_from_rectangle_perimeter (s : ℝ) : 
  s > 0 → 
  (2 * s + 2 * (s / 6) = 30) → 
  (4 * s = 360 / 7) := by
  sorry

#check square_perimeter_from_rectangle_perimeter

end NUMINAMATH_CALUDE_square_perimeter_from_rectangle_perimeter_l1284_128443


namespace NUMINAMATH_CALUDE_yellow_balls_count_l1284_128497

/-- Represents a bag containing red and yellow balls -/
structure BallBag where
  redBalls : ℕ
  yellowBalls : ℕ

/-- Calculates the probability of drawing a red ball from the bag -/
def redProbability (bag : BallBag) : ℚ :=
  bag.redBalls / (bag.redBalls + bag.yellowBalls)

/-- Theorem: Given the conditions, the number of yellow balls is 25 -/
theorem yellow_balls_count (bag : BallBag) 
  (h1 : bag.redBalls = 10)
  (h2 : redProbability bag = 2/7) :
  bag.yellowBalls = 25 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l1284_128497


namespace NUMINAMATH_CALUDE_least_integer_with_remainder_l1284_128461

theorem least_integer_with_remainder (n : ℕ) : n = 842 ↔ 
  (n > 1) ∧
  (∀ d ∈ ({5, 6, 7, 8, 10} : Set ℕ), n % d = 2) ∧
  (∀ m : ℕ, m > 1 → (∀ d ∈ ({5, 6, 7, 8, 10} : Set ℕ), m % d = 2) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_least_integer_with_remainder_l1284_128461


namespace NUMINAMATH_CALUDE_original_manufacturing_cost_l1284_128415

/-- 
Given a fixed selling price and information about profit changes,
prove that the original manufacturing cost was $70.
-/
theorem original_manufacturing_cost
  (P : ℝ) -- Selling price
  (h1 : P - P * 0.5 = 50) -- New manufacturing cost is $50
  : P * 0.7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_original_manufacturing_cost_l1284_128415


namespace NUMINAMATH_CALUDE_chord_length_is_2_sqrt_2_l1284_128403

/-- The circle with center (1, 1) and radius 2 -/
def C : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 = 4}

/-- The line x - y + 2 = 0 -/
def l : Set (ℝ × ℝ) := {p | p.1 - p.2 + 2 = 0}

/-- The length of the chord intercepted by line l on circle C -/
def chord_length : ℝ := sorry

/-- Theorem stating that the chord length is 2√2 -/
theorem chord_length_is_2_sqrt_2 : chord_length = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_chord_length_is_2_sqrt_2_l1284_128403


namespace NUMINAMATH_CALUDE_sin_plus_cos_equivalence_l1284_128480

theorem sin_plus_cos_equivalence (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.cos (3 * x - π / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_equivalence_l1284_128480


namespace NUMINAMATH_CALUDE_paul_clothing_expense_l1284_128429

def shirt_price : ℝ := 15
def pants_price : ℝ := 40
def suit_price : ℝ := 150
def sweater_price : ℝ := 30

def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_suits : ℕ := 1
def num_sweaters : ℕ := 2

def store_discount : ℝ := 0.2
def coupon_discount : ℝ := 0.1

def total_before_discount : ℝ := 
  shirt_price * num_shirts + 
  pants_price * num_pants + 
  suit_price * num_suits + 
  sweater_price * num_sweaters

def total_after_store_discount : ℝ :=
  total_before_discount * (1 - store_discount)

def final_total : ℝ :=
  total_after_store_discount * (1 - coupon_discount)

theorem paul_clothing_expense : final_total = 252 :=
sorry

end NUMINAMATH_CALUDE_paul_clothing_expense_l1284_128429


namespace NUMINAMATH_CALUDE_real_roots_quadratic_equation_l1284_128482

theorem real_roots_quadratic_equation (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ k ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_equation_l1284_128482


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_three_l1284_128479

theorem sum_of_x_and_y_is_three (x y : ℝ) (h : x^2 + y^2 = 14*x - 8*y - 74) : x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_three_l1284_128479


namespace NUMINAMATH_CALUDE_intersection_of_lines_l1284_128410

/-- Given four points in 3D space, this theorem proves that the intersection
    of the lines formed by these points is at a specific coordinate. -/
theorem intersection_of_lines (A B C D : ℝ × ℝ × ℝ) : 
  A = (8, -5, 5) →
  B = (18, -15, 10) →
  C = (1, 5, -7) →
  D = (3, -3, 13) →
  ∃ t s : ℝ, 
    (8 + 10*t, -5 - 10*t, 5 + 5*t) = (1 + 2*s, 5 - 8*s, -7 + 20*s) ∧
    (8 + 10*t, -5 - 10*t, 5 + 5*t) = (-16, 7, -7) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l1284_128410


namespace NUMINAMATH_CALUDE_bug_return_probability_l1284_128478

-- Define the tetrahedron structure
structure Tetrahedron where
  vertices : Fin 4 → Point
  edge_length : ℝ
  is_regular : Bool

-- Define the bug's movement
def bug_move (t : Tetrahedron) (current_vertex : Fin 4) : Fin 4 := sorry

-- Define the probability of returning to the starting vertex after n steps
def return_probability (t : Tetrahedron) (n : ℕ) : ℚ := sorry

-- Main theorem
theorem bug_return_probability (t : Tetrahedron) :
  t.is_regular = true →
  t.edge_length = 1 →
  return_probability t 9 = 4920 / 19683 := by sorry

end NUMINAMATH_CALUDE_bug_return_probability_l1284_128478


namespace NUMINAMATH_CALUDE_train_passengers_with_hats_l1284_128495

theorem train_passengers_with_hats 
  (total_adults : ℕ) 
  (men_percentage : ℚ) 
  (men_with_hats_percentage : ℚ) 
  (women_with_hats_percentage : ℚ) 
  (h1 : total_adults = 3600) 
  (h2 : men_percentage = 40 / 100) 
  (h3 : men_with_hats_percentage = 15 / 100) 
  (h4 : women_with_hats_percentage = 25 / 100) : 
  ℕ := by
  sorry

#check train_passengers_with_hats

end NUMINAMATH_CALUDE_train_passengers_with_hats_l1284_128495


namespace NUMINAMATH_CALUDE_partnership_investment_l1284_128469

/-- Given the investments of partners A and B, the total profit, and A's share of the profit,
    calculate the investment of partner C in a partnership business. -/
theorem partnership_investment (a_invest b_invest total_profit a_profit : ℚ) (h1 : a_invest = 6300)
    (h2 : b_invest = 4200) (h3 : total_profit = 14200) (h4 : a_profit = 4260) :
    ∃ c_invest : ℚ, c_invest = 10500 ∧ 
    a_profit / a_invest = total_profit / (a_invest + b_invest + c_invest) :=
  sorry

end NUMINAMATH_CALUDE_partnership_investment_l1284_128469


namespace NUMINAMATH_CALUDE_sqrt_53_between_consecutive_integers_l1284_128425

theorem sqrt_53_between_consecutive_integers :
  ∃ (n : ℕ), n > 0 ∧ (n : ℝ)^2 < 53 ∧ 53 < (n + 1 : ℝ)^2 ∧ n * (n + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_53_between_consecutive_integers_l1284_128425


namespace NUMINAMATH_CALUDE_solve_average_problem_l1284_128484

def average_problem (total_average : ℚ) (pair1_average : ℚ) (pair2_average : ℚ) (pair3_average : ℚ) : Prop :=
  ∃ (n : ℕ) (sum : ℚ),
    n > 0 ∧
    sum / n = total_average ∧
    n = 6 ∧
    sum = 2 * pair1_average + 2 * pair2_average + 2 * pair3_average

theorem solve_average_problem :
  average_problem (395/100) (38/10) (385/100) (4200000000000001/1000000000000000) :=
sorry

end NUMINAMATH_CALUDE_solve_average_problem_l1284_128484


namespace NUMINAMATH_CALUDE_sum_of_distances_l1284_128428

theorem sum_of_distances (saham_distance mother_distance : ℝ) 
  (h1 : saham_distance = 2.6)
  (h2 : mother_distance = 5.98) :
  saham_distance + mother_distance = 8.58 := by
sorry

end NUMINAMATH_CALUDE_sum_of_distances_l1284_128428


namespace NUMINAMATH_CALUDE_planting_probabilities_l1284_128416

structure CropPlanting where
  transition_A : Fin 2 → ℚ
  transition_B : Fin 2 → ℚ
  transition_C : Fin 2 → ℚ

def planting : CropPlanting :=
  { transition_A := ![1/3, 2/3],
    transition_B := ![1/4, 3/4],
    transition_C := ![2/5, 3/5] }

def probability_A_third_given_B_first (p : CropPlanting) : ℚ :=
  p.transition_B 1 * p.transition_C 0

def distribution_X_given_A_first (p : CropPlanting) : Fin 2 → ℚ
  | 0 => p.transition_A 1 * p.transition_C 1 + p.transition_A 0 * p.transition_B 1
  | 1 => p.transition_A 1 * p.transition_C 0 + p.transition_A 0 * p.transition_B 0

def expectation_X_given_A_first (p : CropPlanting) : ℚ :=
  1 * distribution_X_given_A_first p 0 + 2 * distribution_X_given_A_first p 1

theorem planting_probabilities :
  probability_A_third_given_B_first planting = 3/10 ∧
  distribution_X_given_A_first planting 0 = 13/20 ∧
  distribution_X_given_A_first planting 1 = 7/20 ∧
  expectation_X_given_A_first planting = 27/20 := by
  sorry

end NUMINAMATH_CALUDE_planting_probabilities_l1284_128416


namespace NUMINAMATH_CALUDE_grace_reading_time_l1284_128485

/-- Represents Grace's reading speed in pages per hour -/
def reading_speed (pages : ℕ) (hours : ℕ) : ℚ :=
  pages / hours

/-- Calculates the time needed to read a book given the number of pages and reading speed -/
def time_to_read (pages : ℕ) (speed : ℚ) : ℚ :=
  pages / speed

/-- Theorem stating that it takes 25 hours to read a 250-page book given Grace's reading rate -/
theorem grace_reading_time :
  let initial_pages : ℕ := 200
  let initial_hours : ℕ := 20
  let target_pages : ℕ := 250
  let speed := reading_speed initial_pages initial_hours
  time_to_read target_pages speed = 25 := by
  sorry


end NUMINAMATH_CALUDE_grace_reading_time_l1284_128485


namespace NUMINAMATH_CALUDE_elina_mean_is_92_5_l1284_128464

/-- The set of all test scores -/
def all_scores : Finset ℕ := {78, 85, 88, 91, 92, 95, 96, 99, 101, 103}

/-- The number of Jason's scores -/
def jason_count : ℕ := 6

/-- The number of Elina's scores -/
def elina_count : ℕ := 4

/-- Jason's mean score -/
def jason_mean : ℚ := 93

/-- The sum of all scores -/
def total_sum : ℕ := Finset.sum all_scores id

theorem elina_mean_is_92_5 :
  (total_sum - jason_count * jason_mean) / elina_count = 92.5 := by
  sorry

end NUMINAMATH_CALUDE_elina_mean_is_92_5_l1284_128464


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1284_128427

theorem triangle_perimeter (a b c : ℝ) (ha : a = 10) (hb : b = 7) (hc : c = 5) :
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1284_128427


namespace NUMINAMATH_CALUDE_system_of_equations_l1284_128481

theorem system_of_equations (x y k : ℝ) :
  x - y = k + 2 →
  x + 3 * y = k →
  x + y = 2 →
  k = 1 := by sorry

end NUMINAMATH_CALUDE_system_of_equations_l1284_128481


namespace NUMINAMATH_CALUDE_equation_solution_l1284_128470

theorem equation_solution (x : ℝ) : 
  (8 * x^2 + 150 * x + 3) / (3 * x + 56) = 4 * x + 2 ↔ x = -1.5 ∨ x = -18.5 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1284_128470


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1284_128411

def A : Set ℤ := {-2, -1, 0, 1, 2}

def B : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1284_128411


namespace NUMINAMATH_CALUDE_inequality_proof_l1284_128453

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_prod : a * b * c * d = 1) :
  1 / (b * c + c * d + d * a - 1) + 1 / (a * b + c * d + d * a - 1) + 
  1 / (a * b + b * c + d * a - 1) + 1 / (a * b + b * c + c * d - 1) ≤ 2 ∧
  (1 / (b * c + c * d + d * a - 1) + 1 / (a * b + c * d + d * a - 1) + 
   1 / (a * b + b * c + d * a - 1) + 1 / (a * b + b * c + c * d - 1) = 2 ↔ a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1284_128453


namespace NUMINAMATH_CALUDE_charles_housesitting_rate_l1284_128465

/-- Represents the earnings of Charles from housesitting and dog walking -/
structure Earnings where
  housesitting_rate : ℝ
  dog_walking_rate : ℝ
  housesitting_hours : ℕ
  dogs_walked : ℕ
  total_earnings : ℝ

/-- Theorem stating that given the conditions, Charles earns $15 per hour for housesitting -/
theorem charles_housesitting_rate (e : Earnings) 
  (h1 : e.dog_walking_rate = 22)
  (h2 : e.housesitting_hours = 10)
  (h3 : e.dogs_walked = 3)
  (h4 : e.total_earnings = 216)
  (h5 : e.housesitting_rate * e.housesitting_hours + e.dog_walking_rate * e.dogs_walked = e.total_earnings) :
  e.housesitting_rate = 15 := by
  sorry

end NUMINAMATH_CALUDE_charles_housesitting_rate_l1284_128465


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l1284_128487

-- Define the constants
def regular_rate : ℝ := 12
def regular_hours : ℝ := 40
def overtime_rate_increase : ℝ := 0.75
def total_hours_worked : ℝ := 63.62

-- Define the function to calculate total compensation
def total_compensation : ℝ :=
  let overtime_hours := total_hours_worked - regular_hours
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  let regular_earnings := regular_rate * regular_hours
  let overtime_earnings := overtime_rate * overtime_hours
  regular_earnings + overtime_earnings

-- Theorem statement
theorem bus_driver_compensation :
  total_compensation = 976.02 := by sorry

end NUMINAMATH_CALUDE_bus_driver_compensation_l1284_128487


namespace NUMINAMATH_CALUDE_compound_interest_initial_sum_l1284_128417

/-- Given an initial sum of money P and an annual compound interest rate r,
    if P(1 + r)² = 8880 and P(1 + r)³ = 9261, then P is approximately equal to 8160. -/
theorem compound_interest_initial_sum (P r : ℝ) 
  (h1 : P * (1 + r)^2 = 8880)
  (h2 : P * (1 + r)^3 = 9261) :
  ∃ ε > 0, |P - 8160| < ε :=
sorry

end NUMINAMATH_CALUDE_compound_interest_initial_sum_l1284_128417


namespace NUMINAMATH_CALUDE_square_side_length_l1284_128408

theorem square_side_length (diagonal : ℝ) (h : diagonal = 4) : 
  ∃ side : ℝ, side = 2 * Real.sqrt 2 ∧ side ^ 2 + side ^ 2 = diagonal ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l1284_128408


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1284_128433

-- Define sets A and B
def A : Set ℝ := {x | x > 3}
def B : Set ℝ := {x | x ≤ 4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 3 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1284_128433


namespace NUMINAMATH_CALUDE_exact_time_l1284_128413

/-- Represents the time in minutes after 4:00 --/
def t : ℝ := by sorry

/-- The angle of the minute hand at time t --/
def minute_hand (t : ℝ) : ℝ := 6 * t

/-- The angle of the hour hand at time t --/
def hour_hand (t : ℝ) : ℝ := 120 + 0.5 * t

/-- The condition that the time is between 4:00 and 5:00 --/
axiom time_range : 0 ≤ t ∧ t < 60

/-- The condition that the minute hand is opposite to where the hour hand was 5 minutes ago --/
axiom opposite_hands : 
  |minute_hand (t + 10) - hour_hand (t - 5)| = 180 ∨ 
  |minute_hand (t + 10) - hour_hand (t - 5)| = 540

theorem exact_time : t = 25 := by sorry

end NUMINAMATH_CALUDE_exact_time_l1284_128413


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1284_128457

theorem cyclic_sum_inequality (a b c : ℝ) (n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) (h5 : n ≥ 2) :
  a / (b + c)^(1/n : ℝ) + b / (c + a)^(1/n : ℝ) + c / (a + b)^(1/n : ℝ) ≥ 3 / 2^(1/n : ℝ) :=
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1284_128457


namespace NUMINAMATH_CALUDE_fourth_score_for_average_l1284_128473

/-- Given three exam scores, this theorem proves the required fourth score to achieve a specific average. -/
theorem fourth_score_for_average (score1 score2 score3 target_avg : ℕ) :
  score1 = 87 →
  score2 = 83 →
  score3 = 88 →
  target_avg = 89 →
  ∃ (score4 : ℕ), (score1 + score2 + score3 + score4) / 4 = target_avg ∧ score4 = 98 := by
  sorry

#check fourth_score_for_average

end NUMINAMATH_CALUDE_fourth_score_for_average_l1284_128473


namespace NUMINAMATH_CALUDE_lines_do_not_intersect_l1284_128456

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line2D) : Prop :=
  ∃ c : ℝ, l1.direction = (c * l2.direction.1, c * l2.direction.2)

/-- The first line -/
def line1 : Line2D :=
  { point := (1, 3), direction := (5, -8) }

/-- The second line -/
def line2 (k : ℝ) : Line2D :=
  { point := (-1, 4), direction := (2, k) }

/-- Theorem: The lines do not intersect if and only if k = -16/5 -/
theorem lines_do_not_intersect (k : ℝ) : 
  are_parallel line1 (line2 k) ↔ k = -16/5 := by
  sorry

end NUMINAMATH_CALUDE_lines_do_not_intersect_l1284_128456


namespace NUMINAMATH_CALUDE_vector_linear_combination_l1284_128436

/-- Given vectors a, b, and c in ℝ², prove that c can be expressed as a linear combination of a and b -/
theorem vector_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) 
  (hb : b = (1, -1)) 
  (hc : c = (-1, 2)) : 
  c = (1/2 : ℝ) • a - (3/2 : ℝ) • b :=
sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l1284_128436


namespace NUMINAMATH_CALUDE_inequality_solution_l1284_128420

theorem inequality_solution : ∃! x : ℝ, 
  (Real.sqrt (x^3 + x - 90) + 7) * |x^3 - 10*x^2 + 31*x - 28| ≤ 0 ∧
  x = 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1284_128420


namespace NUMINAMATH_CALUDE_employee_count_l1284_128474

theorem employee_count (avg_salary : ℕ) (salary_increase : ℕ) (manager_salary : ℕ)
  (h1 : avg_salary = 1300)
  (h2 : salary_increase = 100)
  (h3 : manager_salary = 3400) :
  ∃ n : ℕ, n * avg_salary + manager_salary = (n + 1) * (avg_salary + salary_increase) ∧ n = 20 :=
by sorry

end NUMINAMATH_CALUDE_employee_count_l1284_128474


namespace NUMINAMATH_CALUDE_right_angle_vector_coord_l1284_128438

/-- Given two vectors OA and OB in a 2D Cartesian coordinate system, 
    if they form a right angle at B, then the y-coordinate of A is 5. -/
theorem right_angle_vector_coord (t : ℝ) : 
  let OA : ℝ × ℝ := (-1, t)
  let OB : ℝ × ℝ := (2, 2)
  let AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)
  (OB.1 * AB.1 + OB.2 * AB.2 = 0) → t = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_angle_vector_coord_l1284_128438


namespace NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l1284_128444

theorem richmond_tigers_ticket_sales (first_half_sales second_half_sales : ℕ) 
  (h1 : first_half_sales = 3867)
  (h2 : second_half_sales = 5703) :
  first_half_sales + second_half_sales = 9570 := by
  sorry

end NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l1284_128444


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_second_vessel_l1284_128423

theorem alcohol_percentage_in_second_vessel
  (vessel1_capacity : ℝ)
  (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ)
  (total_liquid : ℝ)
  (final_vessel_capacity : ℝ)
  (final_mixture_percentage : ℝ)
  (h1 : vessel1_capacity = 3)
  (h2 : vessel1_alcohol_percentage = 25)
  (h3 : vessel2_capacity = 5)
  (h4 : total_liquid = 8)
  (h5 : final_vessel_capacity = 10)
  (h6 : final_mixture_percentage = 27.5)
  : ∃ (vessel2_alcohol_percentage : ℝ),
    vessel2_alcohol_percentage = 40 ∧
    vessel1_capacity * (vessel1_alcohol_percentage / 100) +
    vessel2_capacity * (vessel2_alcohol_percentage / 100) =
    final_vessel_capacity * (final_mixture_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_percentage_in_second_vessel_l1284_128423


namespace NUMINAMATH_CALUDE_gummy_vitamin_price_l1284_128446

theorem gummy_vitamin_price (discount : ℝ) (coupon : ℝ) (total_cost : ℝ) (num_bottles : ℕ) : 
  discount = 0.20 →
  coupon = 2 →
  total_cost = 30 →
  num_bottles = 3 →
  ∃ (original_price : ℝ), 
    num_bottles * (original_price * (1 - discount) - coupon) = total_cost ∧
    original_price = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_gummy_vitamin_price_l1284_128446


namespace NUMINAMATH_CALUDE_lollipop_difference_l1284_128486

theorem lollipop_difference (henry alison diane : ℕ) : 
  henry > alison →
  alison = 60 →
  alison = diane / 2 →
  henry + alison + diane = 45 * 6 →
  henry - alison = 30 := by
sorry

end NUMINAMATH_CALUDE_lollipop_difference_l1284_128486


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1284_128489

theorem inequality_solution_set (x : ℝ) :
  (1 / (x + 2) + 5 / (x + 4) ≤ 1) ↔ (x ≤ -4 ∨ x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1284_128489


namespace NUMINAMATH_CALUDE_correct_years_until_twice_as_old_l1284_128496

/-- Represents the current ages of the three brothers -/
structure BrothersAges where
  david : ℕ
  richard : ℕ
  scott : ℕ

/-- Calculates the number of years until Richard is twice as old as Scott -/
def yearsUntilTwiceAsOld (ages : BrothersAges) : ℕ :=
  sorry

theorem correct_years_until_twice_as_old : 
  ∀ (ages : BrothersAges),
    ages.david = 14 →
    ages.richard = ages.david + 6 →
    ages.scott = ages.david - 8 →
    yearsUntilTwiceAsOld ages = 8 :=
  sorry

end NUMINAMATH_CALUDE_correct_years_until_twice_as_old_l1284_128496


namespace NUMINAMATH_CALUDE_g_evaluation_l1284_128472

def g (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 7

theorem g_evaluation : 3 * g 2 + 4 * g (-2) = 113 := by
  sorry

end NUMINAMATH_CALUDE_g_evaluation_l1284_128472


namespace NUMINAMATH_CALUDE_red_tetrahedron_volume_l1284_128452

/-- The volume of a tetrahedron formed by red vertices in a cube with alternately colored vertices --/
theorem red_tetrahedron_volume (cube_side_length : ℝ) (h : cube_side_length = 8) :
  let cube_volume := cube_side_length ^ 3
  let blue_tetrahedron_volume := (1 / 3) * cube_side_length ^ 3 / 2
  let red_tetrahedron_volume := cube_volume - 4 * blue_tetrahedron_volume
  red_tetrahedron_volume = 512 - (4 * 256 / 3) := by
  sorry

#eval 512 - (4 * 256 / 3)  -- To verify the numerical result

end NUMINAMATH_CALUDE_red_tetrahedron_volume_l1284_128452


namespace NUMINAMATH_CALUDE_max_quotient_value_l1284_128448

theorem max_quotient_value (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) :
  (∀ x y, 300 ≤ x ∧ x ≤ 500 ∧ 800 ≤ y ∧ y ≤ 1600 → y / x ≤ 16 / 3) ∧
  (∃ x y, 300 ≤ x ∧ x ≤ 500 ∧ 800 ≤ y ∧ y ≤ 1600 ∧ y / x = 16 / 3) :=
sorry

end NUMINAMATH_CALUDE_max_quotient_value_l1284_128448


namespace NUMINAMATH_CALUDE_divisor_problem_l1284_128418

/-- 
Given a dividend of 23, a quotient of 5, and a remainder of 3, 
prove that the divisor is 4.
-/
theorem divisor_problem (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) : 
  dividend = 23 → quotient = 5 → remainder = 3 → 
  dividend = divisor * quotient + remainder → 
  divisor = 4 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l1284_128418


namespace NUMINAMATH_CALUDE_marias_cookies_l1284_128402

theorem marias_cookies (x : ℕ) : 
  x ≥ 5 →
  (x - 5) % 2 = 0 →
  ((x - 5) / 2 - 2 = 5) →
  x = 19 := by
sorry

end NUMINAMATH_CALUDE_marias_cookies_l1284_128402


namespace NUMINAMATH_CALUDE_m_range_l1284_128454

theorem m_range (m : ℝ) (h1 : m < 0) (h2 : ∀ x : ℝ, x^2 + m*x + 1 > 0) : -2 < m ∧ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l1284_128454


namespace NUMINAMATH_CALUDE_periodic_odd_function_at_six_l1284_128442

/-- An odd function that satisfies f(x+2) = -f(x) for all x -/
def periodic_odd_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2) = -f x)

/-- For a periodic odd function f, f(6) = 0 -/
theorem periodic_odd_function_at_six (f : ℝ → ℝ) (h : periodic_odd_function f) : f 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_periodic_odd_function_at_six_l1284_128442


namespace NUMINAMATH_CALUDE_lcm_gcd_product_equals_product_l1284_128475

theorem lcm_gcd_product_equals_product (a b : ℕ) (ha : a = 12) (hb : b = 18) :
  (Nat.lcm a b) * (Nat.gcd a b) = a * b := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_equals_product_l1284_128475


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l1284_128421

/-- The rowing speed of a man in still water, given his speeds with and against a stream -/
theorem mans_rowing_speed (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 16)
  (h2 : speed_against_stream = 4) : 
  (speed_with_stream + speed_against_stream) / 2 = 10 := by
  sorry

#check mans_rowing_speed

end NUMINAMATH_CALUDE_mans_rowing_speed_l1284_128421


namespace NUMINAMATH_CALUDE_positive_inequality_l1284_128463

theorem positive_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^3 + b^3 = 1) :
  a^2 + a*b + b^2 - a - b > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_inequality_l1284_128463


namespace NUMINAMATH_CALUDE_original_bet_is_40_l1284_128459

/-- Represents the payout ratio for a blackjack -/
def blackjack_ratio : ℚ := 3 / 2

/-- Represents the payout received by the player -/
def payout : ℚ := 60

/-- Calculates the original bet given the payout and the blackjack ratio -/
def original_bet (payout : ℚ) (ratio : ℚ) : ℚ := payout / ratio

/-- Proves that the original bet was $40 given the conditions -/
theorem original_bet_is_40 : 
  original_bet payout blackjack_ratio = 40 := by
  sorry

#eval original_bet payout blackjack_ratio

end NUMINAMATH_CALUDE_original_bet_is_40_l1284_128459


namespace NUMINAMATH_CALUDE_max_log_value_l1284_128494

theorem max_log_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 4 * a - 2 * b + 25 * c = 0) : 
  (∀ x y z, x > 0 → y > 0 → z > 0 → 4 * x - 2 * y + 25 * z = 0 → 
    Real.log x + Real.log z - 2 * Real.log y ≤ Real.log a + Real.log c - 2 * Real.log b) ∧
  Real.log a + Real.log c - 2 * Real.log b = -2 * Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_max_log_value_l1284_128494


namespace NUMINAMATH_CALUDE_pascal_cycling_trip_l1284_128430

theorem pascal_cycling_trip (current_speed : ℝ) (speed_reduction : ℝ) (time_increase : ℝ) 
  (h1 : current_speed = 8)
  (h2 : speed_reduction = 4)
  (h3 : time_increase = 16)
  (h4 : current_speed * (time_increase + t) = (current_speed - speed_reduction) * (time_increase + t + time_increase))
  (h5 : current_speed * t = (current_speed + current_speed / 2) * (time_increase + t - time_increase)) :
  current_speed * t = 256 := by
  sorry

end NUMINAMATH_CALUDE_pascal_cycling_trip_l1284_128430


namespace NUMINAMATH_CALUDE_Betty_wallet_contribution_ratio_l1284_128419

theorem Betty_wallet_contribution_ratio :
  let wallet_cost : ℚ := 100
  let initial_savings : ℚ := wallet_cost / 2
  let parents_contribution : ℚ := 15
  let remaining_need : ℚ := 5
  let grandparents_contribution : ℚ := wallet_cost - initial_savings - parents_contribution - remaining_need
  grandparents_contribution / parents_contribution = 2 := by
    sorry

end NUMINAMATH_CALUDE_Betty_wallet_contribution_ratio_l1284_128419


namespace NUMINAMATH_CALUDE_fencing_cost_theorem_l1284_128498

/-- The total cost of fencing a rectangular plot -/
def fencing_cost (length breadth cost_per_metre : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_metre

/-- Theorem: The total cost of fencing a rectangular plot with given dimensions -/
theorem fencing_cost_theorem (length breadth cost_per_metre : ℝ) 
  (h1 : length = 60)
  (h2 : breadth = length - 20)
  (h3 : cost_per_metre = 26.50) :
  fencing_cost length breadth cost_per_metre = 5300 := by
  sorry

#eval fencing_cost 60 40 26.50

end NUMINAMATH_CALUDE_fencing_cost_theorem_l1284_128498


namespace NUMINAMATH_CALUDE_regular_pentagon_side_length_l1284_128424

/-- A regular pentagon with a perimeter of 23.4 cm has sides of length 4.68 cm. -/
theorem regular_pentagon_side_length : 
  ∀ (p : ℝ) (s : ℝ), 
  p = 23.4 →  -- perimeter is 23.4 cm
  s = p / 5 →  -- side length is perimeter divided by 5 (number of sides in a pentagon)
  s = 4.68 := by
sorry

end NUMINAMATH_CALUDE_regular_pentagon_side_length_l1284_128424


namespace NUMINAMATH_CALUDE_integral_equals_antiderivative_l1284_128447

open Real

noncomputable def f (x : ℝ) : ℝ := (x^3 - 6*x^2 + 13*x - 8) / (x*(x-2)^3)

noncomputable def F (x : ℝ) : ℝ := log (abs x) - 1 / (2*(x-2)^2)

theorem integral_equals_antiderivative (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 2) :
  deriv F x = f x :=
by sorry

end NUMINAMATH_CALUDE_integral_equals_antiderivative_l1284_128447


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1284_128405

/-- The coefficient of x^3y^7 in the expansion of (2/3x - 3/4y)^10 -/
def coefficient_x3y7 : ℚ :=
  let a : ℚ := 2/3
  let b : ℚ := -3/4
  let n : ℕ := 10
  let k : ℕ := 7
  (n.choose k) * a^(n-k) * b^k

theorem expansion_coefficient :
  coefficient_x3y7 = -4374/921 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1284_128405


namespace NUMINAMATH_CALUDE_vector_subtraction_l1284_128422

/-- Given two plane vectors a and b, prove that a - 2b equals (3, 7) -/
theorem vector_subtraction (a b : ℝ × ℝ) (ha : a = (5, 3)) (hb : b = (1, -2)) :
  a - 2 • b = (3, 7) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1284_128422


namespace NUMINAMATH_CALUDE_total_tax_percentage_calculation_l1284_128412

/-- Calculates the total tax percentage given spending percentages and tax rates -/
def totalTaxPercentage (clothingSpendPercentage : ℝ) (foodSpendPercentage : ℝ) 
  (electronicsSpendPercentage : ℝ) (otherSpendPercentage : ℝ)
  (clothingTaxRate : ℝ) (foodTaxRate : ℝ) (electronicsTaxRate : ℝ) (otherTaxRate : ℝ) : ℝ :=
  clothingSpendPercentage * clothingTaxRate + 
  foodSpendPercentage * foodTaxRate + 
  electronicsSpendPercentage * electronicsTaxRate + 
  otherSpendPercentage * otherTaxRate

theorem total_tax_percentage_calculation :
  totalTaxPercentage 0.585 0.12 0.225 0.07 0.052 0 0.073 0.095 = 0.053495 := by
  sorry

end NUMINAMATH_CALUDE_total_tax_percentage_calculation_l1284_128412


namespace NUMINAMATH_CALUDE_system_solution_square_difference_l1284_128439

theorem system_solution_square_difference (x y : ℝ) 
  (eq1 : 3 * x - 2 * y = 1) 
  (eq2 : x + y = 2) : 
  x^2 - 2 * y^2 = -1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_square_difference_l1284_128439


namespace NUMINAMATH_CALUDE_largest_average_l1284_128432

def multiples_average (m n : ℕ) : ℚ :=
  (m + n * (n.div m) * m) / (2 * n.div m)

theorem largest_average : 
  let avg3 := multiples_average 3 101
  let avg4 := multiples_average 4 102
  let avg5 := multiples_average 5 100
  let avg7 := multiples_average 7 101
  avg5 = 52.5 ∧ avg7 = 52.5 ∧ avg5 > avg3 ∧ avg5 > avg4 :=
by sorry

end NUMINAMATH_CALUDE_largest_average_l1284_128432


namespace NUMINAMATH_CALUDE_min_sqrt_difference_l1284_128462

theorem min_sqrt_difference (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ (m n : ℕ), 
    0 < m ∧ 0 < n ∧ m ≤ n ∧
    (∀ (a b : ℕ), 0 < a → 0 < b → a ≤ b → 
      Real.sqrt (2 * p) - Real.sqrt m - Real.sqrt n ≤ 
      Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b) ∧
    m = (p - 1) / 2 ∧ n = (p + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_sqrt_difference_l1284_128462


namespace NUMINAMATH_CALUDE_ratio_of_sums_l1284_128426

/-- An arithmetic sequence with common difference d, first term 8d, and sum of first n terms S_n -/
structure ArithmeticSequence (d : ℝ) where
  a : ℕ → ℝ
  S : ℕ → ℝ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d
  h3 : a 1 = 8 * d
  h4 : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The ratio of 7S_5 to 5S_7 is 10/11 for the given arithmetic sequence -/
theorem ratio_of_sums (d : ℝ) (seq : ArithmeticSequence d) :
  7 * seq.S 5 / (5 * seq.S 7) = 10 / 11 :=
sorry

end NUMINAMATH_CALUDE_ratio_of_sums_l1284_128426


namespace NUMINAMATH_CALUDE_sin_300_degrees_l1284_128434

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l1284_128434


namespace NUMINAMATH_CALUDE_yaras_earnings_l1284_128492

/-- Yara's work and earnings over two weeks -/
theorem yaras_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) 
  (h1 : hours_week1 = 12)
  (h2 : hours_week2 = 18)
  (h3 : extra_earnings = 36)
  (h4 : ∃ (wage : ℚ), wage * (hours_week2 - hours_week1) = extra_earnings) :
  ∃ (total_earnings : ℚ), total_earnings = hours_week1 * (extra_earnings / (hours_week2 - hours_week1)) + 
                           hours_week2 * (extra_earnings / (hours_week2 - hours_week1)) ∧
                           total_earnings = 180 := by
  sorry


end NUMINAMATH_CALUDE_yaras_earnings_l1284_128492


namespace NUMINAMATH_CALUDE_marian_cookies_l1284_128451

theorem marian_cookies (cookies_per_tray : ℕ) (num_trays : ℕ) (h1 : cookies_per_tray = 12) (h2 : num_trays = 23) :
  cookies_per_tray * num_trays = 276 := by
  sorry

end NUMINAMATH_CALUDE_marian_cookies_l1284_128451


namespace NUMINAMATH_CALUDE_angle_measures_in_special_cyclic_quadrilateral_l1284_128450

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral :=
  (A B C D : ℝ)
  (cyclic : A + C = 180 ∧ B + D = 180)

-- Define the diagonal property
def DiagonalProperty (q : CyclicQuadrilateral) :=
  ∃ (θ : ℝ), (q.A = 6 * θ ∨ q.C = 6 * θ) ∧ (q.B = 6 * θ ∨ q.D = 6 * θ)

-- Define the set of possible angle measures
def PossibleAngleMeasures : Set ℝ := {45, 135, 225/2, 135/2}

-- Theorem statement
theorem angle_measures_in_special_cyclic_quadrilateral
  (q : CyclicQuadrilateral) (h : DiagonalProperty q) :
  q.A ∈ PossibleAngleMeasures :=
sorry

end NUMINAMATH_CALUDE_angle_measures_in_special_cyclic_quadrilateral_l1284_128450


namespace NUMINAMATH_CALUDE_ethereum_investment_l1284_128488

theorem ethereum_investment (I : ℝ) : 
  I > 0 →
  (I * 1.25 * 1.5 = 750) →
  I = 400 := by
sorry

end NUMINAMATH_CALUDE_ethereum_investment_l1284_128488


namespace NUMINAMATH_CALUDE_winter_clothing_count_l1284_128406

theorem winter_clothing_count (num_boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ) :
  num_boxes = 6 →
  scarves_per_box = 5 →
  mittens_per_box = 5 →
  num_boxes * (scarves_per_box + mittens_per_box) = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_winter_clothing_count_l1284_128406


namespace NUMINAMATH_CALUDE_division_problem_l1284_128466

theorem division_problem (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 100 →
  divisor = 11 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 1 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1284_128466


namespace NUMINAMATH_CALUDE_max_profit_l1284_128400

-- Define the types of products
inductive Product
| A
| B

-- Define the profit function
def profit (x y : ℕ) : ℕ := 300 * x + 400 * y

-- Define the material constraints
def material_constraint (x y : ℕ) : Prop :=
  x + 2 * y ≤ 12 ∧ 2 * x + y ≤ 12

-- State the theorem
theorem max_profit :
  ∃ x y : ℕ,
    material_constraint x y ∧
    profit x y = 2800 ∧
    ∀ a b : ℕ, material_constraint a b → profit a b ≤ 2800 :=
sorry

end NUMINAMATH_CALUDE_max_profit_l1284_128400


namespace NUMINAMATH_CALUDE_coefficients_of_specific_quadratic_l1284_128491

/-- Given a quadratic equation ax^2 + bx + c = 0, this function returns the tuple (a, b, c) of its coefficients -/
def quadratic_coefficients (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

/-- The coefficients of the quadratic equation x^2 - x + 3 = 0 are (1, -1, 3) -/
theorem coefficients_of_specific_quadratic :
  quadratic_coefficients 1 (-1) 3 = (1, -1, 3) := by
  sorry

end NUMINAMATH_CALUDE_coefficients_of_specific_quadratic_l1284_128491


namespace NUMINAMATH_CALUDE_part1_part2_part3_l1284_128483

-- Define the quadratic function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + 1

-- Part 1
theorem part1 (a : ℝ) (h1 : a ≠ 0) :
  (∀ x, f a x > 0 ↔ -1/3 < x ∧ x < 1/2) → a = 1/5 := by sorry

-- Part 2
theorem part2 (a : ℝ) (h2 : a ∈ Set.Icc (-2 : ℝ) 0) :
  {x : ℝ | f a x > 0} = {x : ℝ | -1/2 < x ∧ x < 1} := by sorry

-- Part 3
theorem part3 (a : ℝ) :
  (∀ x ∈ Set.Icc 0 2, f a x > 0) → a > -3/4 ∧ a ≠ 0 := by sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l1284_128483


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_4sqrt6_l1284_128460

theorem sqrt_sum_equals_4sqrt6 :
  Real.sqrt (16 - 12 * Real.sqrt 3) + Real.sqrt (16 + 12 * Real.sqrt 3) = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_4sqrt6_l1284_128460


namespace NUMINAMATH_CALUDE_upgraded_fraction_is_one_fourth_l1284_128407

/-- Represents a satellite with modular units and sensors -/
structure Satellite where
  units : ℕ
  non_upgraded_per_unit : ℕ
  upgraded_total : ℕ

/-- The fraction of upgraded sensors on a satellite -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.upgraded_total / (s.units * s.non_upgraded_per_unit + s.upgraded_total)

/-- Theorem: The fraction of upgraded sensors is 1/4 under given conditions -/
theorem upgraded_fraction_is_one_fourth (s : Satellite) 
    (h1 : s.units = 24)
    (h2 : s.non_upgraded_per_unit = s.upgraded_total / 8) :
  upgraded_fraction s = 1/4 := by
  sorry

#eval upgraded_fraction { units := 24, non_upgraded_per_unit := 1, upgraded_total := 8 }

end NUMINAMATH_CALUDE_upgraded_fraction_is_one_fourth_l1284_128407
