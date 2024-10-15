import Mathlib

namespace NUMINAMATH_CALUDE_tangent_at_2_minus_6_tangent_through_origin_l2309_230956

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem for the tangent line at (2, -6)
theorem tangent_at_2_minus_6 :
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 13 * x - 32 :=
sorry

-- Theorem for the tangent line passing through the origin
theorem tangent_through_origin :
  ∃ x₀ y₀ : ℝ,
    f x₀ = y₀ ∧
    f' x₀ * (-x₀) + y₀ = 0 ∧
    (∀ x y : ℝ, y = f' x₀ * x ↔ y = 13 * x) ∧
    x₀ = -2 ∧
    y₀ = -26 :=
sorry

end NUMINAMATH_CALUDE_tangent_at_2_minus_6_tangent_through_origin_l2309_230956


namespace NUMINAMATH_CALUDE_number_of_walls_l2309_230941

/-- Given the following conditions:
  - Each wall has 30 bricks in a single row
  - There are 50 rows in each wall
  - 3000 bricks will be used to make all the walls
  Prove that the number of walls that can be built is 2. -/
theorem number_of_walls (bricks_per_row : ℕ) (rows_per_wall : ℕ) (total_bricks : ℕ) :
  bricks_per_row = 30 →
  rows_per_wall = 50 →
  total_bricks = 3000 →
  total_bricks / (bricks_per_row * rows_per_wall) = 2 :=
by sorry

end NUMINAMATH_CALUDE_number_of_walls_l2309_230941


namespace NUMINAMATH_CALUDE_evaluate_expression_l2309_230914

theorem evaluate_expression (a x : ℝ) (h : x = 2 * a + 6) :
  2 * (x - a + 5) = 2 * a + 22 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2309_230914


namespace NUMINAMATH_CALUDE_smallest_repeating_block_of_nine_elevenths_l2309_230954

/-- The number of digits in the smallest repeating block of the decimal expansion of 9/11 -/
def smallest_repeating_block_length : ℕ :=
  2

/-- The fraction we're considering -/
def fraction : ℚ :=
  9 / 11

theorem smallest_repeating_block_of_nine_elevenths :
  smallest_repeating_block_length = 2 ∧
  ∃ (a b : ℕ) (k : ℕ+), fraction = (a * 10^smallest_repeating_block_length + b) / (10^smallest_repeating_block_length - 1) / k :=
by sorry

end NUMINAMATH_CALUDE_smallest_repeating_block_of_nine_elevenths_l2309_230954


namespace NUMINAMATH_CALUDE_range_of_a_l2309_230979

-- Define the circle M
def circle_M (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*a*x - 2*a*y = 0

-- Define point A
def point_A : ℝ × ℝ := (0, 2)

-- Define the condition that A is outside the circle M
def A_outside_M (a : ℝ) : Prop :=
  ∀ x y, circle_M a x y → (x - 0)^2 + (y - 2)^2 > (x - a)^2 + (y - a)^2

-- Define the existence of point T
def exists_T (a : ℝ) : Prop :=
  ∃ x y, circle_M a x y ∧ 
    Real.cos (Real.pi/4) * (x - 0) + Real.sin (Real.pi/4) * (y - 2) = 
    Real.sqrt ((x - 0)^2 + (y - 2)^2) * Real.sqrt ((x - a)^2 + (y - a)^2)

-- Theorem statement
theorem range_of_a : 
  ∀ a : ℝ, a > 0 → A_outside_M a → exists_T a → Real.sqrt 3 - 1 ≤ a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2309_230979


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l2309_230963

theorem unique_solution_power_equation :
  ∃! (x y z t : ℕ+), 2^y.val + 2^z.val * 5^t.val - 5^x.val = 1 ∧
    x = 2 ∧ y = 4 ∧ z = 1 ∧ t = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l2309_230963


namespace NUMINAMATH_CALUDE_scientific_notation_86560_l2309_230997

theorem scientific_notation_86560 : 
  86560 = 8.656 * (10 ^ 4) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_86560_l2309_230997


namespace NUMINAMATH_CALUDE_raisin_difference_is_twenty_l2309_230968

/-- The number of raisin cookies Helen baked yesterday -/
def yesterday_raisin : ℕ := 300

/-- The number of raisin cookies Helen baked today -/
def today_raisin : ℕ := 280

/-- The number of chocolate chip cookies Helen baked yesterday -/
def yesterday_chocolate : ℕ := 519

/-- The number of chocolate chip cookies Helen baked today -/
def today_chocolate : ℕ := 359

/-- The difference in raisin cookies baked between yesterday and today -/
def raisin_difference : ℕ := yesterday_raisin - today_raisin

theorem raisin_difference_is_twenty : raisin_difference = 20 := by
  sorry

end NUMINAMATH_CALUDE_raisin_difference_is_twenty_l2309_230968


namespace NUMINAMATH_CALUDE_value_of_M_l2309_230952

theorem value_of_M : ∃ M : ℝ, (0.3 * M = 0.6 * 500) ∧ (M = 1000) := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l2309_230952


namespace NUMINAMATH_CALUDE_soccer_team_selection_l2309_230935

/-- The total number of players in the soccer team -/
def total_players : ℕ := 16

/-- The number of quadruplets in the team -/
def num_quadruplets : ℕ := 4

/-- The number of players to be chosen as starters -/
def num_starters : ℕ := 6

/-- The number of quadruplets to be chosen as starters -/
def num_quadruplets_chosen : ℕ := 1

/-- The number of ways to choose the starting lineup -/
def num_ways : ℕ := 3168

theorem soccer_team_selection :
  (num_quadruplets * Nat.choose (total_players - num_quadruplets) (num_starters - num_quadruplets_chosen)) = num_ways :=
sorry

end NUMINAMATH_CALUDE_soccer_team_selection_l2309_230935


namespace NUMINAMATH_CALUDE_solve_for_x_l2309_230973

theorem solve_for_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 14) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2309_230973


namespace NUMINAMATH_CALUDE_complement_union_equals_set_l2309_230939

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {2,4,5}
def B : Set Nat := {1,2,5}

theorem complement_union_equals_set : 
  (U \ (A ∪ B)) = {3,6} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_set_l2309_230939


namespace NUMINAMATH_CALUDE_hurdle_race_calculations_l2309_230980

/-- Calculates the distance between adjacent hurdles and the theoretical best time for a 110m hurdle race --/
theorem hurdle_race_calculations 
  (total_distance : ℝ) 
  (num_hurdles : ℕ) 
  (start_to_first : ℝ) 
  (last_to_finish : ℝ) 
  (time_to_first : ℝ) 
  (time_after_last : ℝ) 
  (fastest_cycle : ℝ) 
  (h1 : total_distance = 110) 
  (h2 : num_hurdles = 10) 
  (h3 : start_to_first = 13.72) 
  (h4 : last_to_finish = 14.02) 
  (h5 : time_to_first = 2.5) 
  (h6 : time_after_last = 1.4) 
  (h7 : fastest_cycle = 0.96) :
  let inter_hurdle_distance := (total_distance - start_to_first - last_to_finish) / num_hurdles
  let theoretical_best_time := time_to_first + (num_hurdles : ℝ) * fastest_cycle + time_after_last
  inter_hurdle_distance = 8.28 ∧ theoretical_best_time = 12.1 := by
  sorry


end NUMINAMATH_CALUDE_hurdle_race_calculations_l2309_230980


namespace NUMINAMATH_CALUDE_prob_four_successes_in_five_trials_l2309_230903

/-- The probability of exactly 4 successes in 5 independent Bernoulli trials with p = 1/3 -/
theorem prob_four_successes_in_five_trials : 
  let n : ℕ := 5
  let p : ℝ := 1/3
  let k : ℕ := 4
  Nat.choose n k * p^k * (1-p)^(n-k) = 10/243 := by
  sorry

end NUMINAMATH_CALUDE_prob_four_successes_in_five_trials_l2309_230903


namespace NUMINAMATH_CALUDE_point_equal_distance_to_axes_l2309_230989

/-- A point P with coordinates (m-4, 2m+7) has equal distance from both coordinate axes if and only if m = -11 or m = -1 -/
theorem point_equal_distance_to_axes (m : ℝ) : 
  |m - 4| = |2*m + 7| ↔ m = -11 ∨ m = -1 := by
sorry

end NUMINAMATH_CALUDE_point_equal_distance_to_axes_l2309_230989


namespace NUMINAMATH_CALUDE_exactly_four_separators_l2309_230962

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of five points in a plane -/
def FivePointSet := Fin 5 → Point

/-- A circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if three points are collinear -/
def are_collinear (p q r : Point) : Prop := sorry

/-- Predicate to check if four points are concyclic -/
def are_concyclic (p q r s : Point) : Prop := sorry

/-- Predicate to check if a point is inside a circle -/
def is_inside (p : Point) (c : Circle) : Prop := sorry

/-- Predicate to check if a point is outside a circle -/
def is_outside (p : Point) (c : Circle) : Prop := sorry

/-- Predicate to check if a point is on a circle -/
def is_on_circle (p : Point) (c : Circle) : Prop := sorry

/-- Predicate to check if a circle is a separator for a set of five points -/
def is_separator (c : Circle) (s : FivePointSet) : Prop :=
  ∃ (i j k l m : Fin 5),
    i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ m ≠ i ∧ m ≠ j ∧ m ≠ k ∧ m ≠ l ∧
    is_on_circle (s i) c ∧ is_on_circle (s j) c ∧ is_on_circle (s k) c ∧
    is_inside (s l) c ∧ is_outside (s m) c

/-- The main theorem -/
theorem exactly_four_separators (s : FivePointSet) :
  (∀ (i j k : Fin 5), i ≠ j → j ≠ k → k ≠ i → ¬are_collinear (s i) (s j) (s k)) →
  (∀ (i j k l : Fin 5), i ≠ j → j ≠ k → k ≠ l → l ≠ i → ¬are_concyclic (s i) (s j) (s k) (s l)) →
  ∃! (separators : Finset Circle), (∀ c ∈ separators, is_separator c s) ∧ separators.card = 4 :=
sorry

end NUMINAMATH_CALUDE_exactly_four_separators_l2309_230962


namespace NUMINAMATH_CALUDE_william_claire_game_bounds_l2309_230901

/-- A move in William's strategy -/
inductive Move
| reciprocal : Move  -- Replace y with 1/y
| increment  : Move  -- Replace y with y+1

/-- William's strategy for rearranging the numbers -/
def Strategy := List Move

/-- The result of applying a strategy to a sequence of numbers -/
def applyStrategy (s : Strategy) (xs : List ℝ) : List ℝ := sorry

/-- Predicate to check if a list is strictly increasing -/
def isStrictlyIncreasing (xs : List ℝ) : Prop := sorry

/-- The theorem to be proved -/
theorem william_claire_game_bounds :
  ∃ (A B : ℝ) (hA : A > 0) (hB : B > 0),
    ∀ (n : ℕ) (hn : n > 1),
      -- Part (a): William can always succeed in at most An log n moves
      (∀ (xs : List ℝ) (hxs : xs.length = n) (hdistinct : xs.Nodup),
        ∃ (s : Strategy),
          isStrictlyIncreasing (applyStrategy s xs) ∧
          s.length ≤ A * n * Real.log n) ∧
      -- Part (b): Claire can force William to use at least Bn log n moves
      (∃ (xs : List ℝ) (hxs : xs.length = n) (hdistinct : xs.Nodup),
        ∀ (s : Strategy),
          isStrictlyIncreasing (applyStrategy s xs) →
          s.length ≥ B * n * Real.log n) :=
sorry

end NUMINAMATH_CALUDE_william_claire_game_bounds_l2309_230901


namespace NUMINAMATH_CALUDE_area_of_RQST_l2309_230978

/-- Square with side length 3 -/
def Square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

/-- Points on the square -/
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (3, 3)
def D : ℝ × ℝ := (0, 3)

/-- Points E and F divide AB into three segments with ratios 1:2 -/
def E : ℝ × ℝ := (1, 0)
def F : ℝ × ℝ := (2, 0)

/-- Points G and H divide CD similarly -/
def G : ℝ × ℝ := (3, 1)
def H : ℝ × ℝ := (3, 2)

/-- S is the midpoint of AB -/
def S : ℝ × ℝ := (1.5, 0)

/-- Q is the midpoint of CD -/
def Q : ℝ × ℝ := (3, 1.5)

/-- R and T divide the square into two equal areas -/
def R : ℝ × ℝ := (0, 1.5)
def T : ℝ × ℝ := (3, 1.5)

/-- Area of a quadrilateral given its vertices -/
def quadrilateralArea (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  0.5 * abs (p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p4.2 + p4.1 * p1.2
           - (p1.2 * p2.1 + p2.2 * p3.1 + p3.2 * p4.1 + p4.2 * p1.1))

theorem area_of_RQST :
  quadrilateralArea R Q S T = 1.125 := by
  sorry

end NUMINAMATH_CALUDE_area_of_RQST_l2309_230978


namespace NUMINAMATH_CALUDE_find_n_l2309_230947

theorem find_n : ∃ n : ℤ, 3^3 - 7 = 4^2 + 2 + n ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l2309_230947


namespace NUMINAMATH_CALUDE_parallel_and_perpendicular_lines_l2309_230992

-- Define a line in a plane
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a point in a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the concept of a point not being on a line
def PointNotOnLine (p : Point) (l : Line) : Prop :=
  p.y ≠ l.slope * p.x + l.intercept

-- Define the concept of parallel lines
def Parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope ∧ l1.intercept ≠ l2.intercept

-- Define the concept of perpendicular lines
def Perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem parallel_and_perpendicular_lines
  (L : Line) (P : Point) (h : PointNotOnLine P L) :
  (∃! l : Line, Parallel l L ∧ l.slope * P.x + l.intercept = P.y) ∧
  (∃ f : ℝ → Line, ∀ t : ℝ, Perpendicular (f t) L ∧ (f t).slope * P.x + (f t).intercept = P.y) :=
sorry

end NUMINAMATH_CALUDE_parallel_and_perpendicular_lines_l2309_230992


namespace NUMINAMATH_CALUDE_tan_is_odd_l2309_230934

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- State the theorem
theorem tan_is_odd : ∀ x : ℝ, tan (-x) = -tan x := by sorry

end NUMINAMATH_CALUDE_tan_is_odd_l2309_230934


namespace NUMINAMATH_CALUDE_merchant_profit_l2309_230920

theorem merchant_profit (cost_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : 
  markup_percent = 40 →
  discount_percent = 10 →
  let marked_price := cost_price * (1 + markup_percent / 100)
  let selling_price := marked_price * (1 - discount_percent / 100)
  let profit := selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  profit_percent = 26 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_l2309_230920


namespace NUMINAMATH_CALUDE_replaced_person_weight_l2309_230943

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (new_person_weight : ℝ) (average_increase : ℝ) : ℝ :=
  new_person_weight - (initial_count * average_increase)

/-- Theorem stating the weight of the replaced person under given conditions -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 97 4 = 65 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l2309_230943


namespace NUMINAMATH_CALUDE_specific_polygon_area_l2309_230931

/-- A polygon on a grid where dots are spaced one unit apart both horizontally and vertically -/
structure GridPolygon where
  vertices : List (ℤ × ℤ)

/-- Calculate the area of a GridPolygon -/
def area (p : GridPolygon) : ℚ :=
  sorry

/-- The specific polygon described in the problem -/
def specificPolygon : GridPolygon :=
  { vertices := [
    (0, 0), (10, 0), (20, 10), (30, 0), (40, 0),
    (30, 10), (30, 20), (20, 30), (20, 40), (10, 40),
    (0, 40), (0, 10)
  ] }

/-- Theorem stating that the area of the specific polygon is 31.5 square units -/
theorem specific_polygon_area :
  area specificPolygon = 31.5 := by sorry

end NUMINAMATH_CALUDE_specific_polygon_area_l2309_230931


namespace NUMINAMATH_CALUDE_square_difference_ratio_l2309_230907

theorem square_difference_ratio : (2045^2 - 2030^2) / (2050^2 - 2025^2) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_ratio_l2309_230907


namespace NUMINAMATH_CALUDE_smallest_three_digit_solution_l2309_230948

theorem smallest_three_digit_solution :
  ∃ (n : ℕ), 
    n ≥ 100 ∧ 
    n < 1000 ∧ 
    77 * n ≡ 231 [MOD 385] ∧ 
    (∀ m : ℕ, m ≥ 100 ∧ m < n ∧ 77 * m ≡ 231 [MOD 385] → false) ∧
    n = 113 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_solution_l2309_230948


namespace NUMINAMATH_CALUDE_sum_of_absolute_differences_l2309_230929

theorem sum_of_absolute_differences (a b c : ℤ) 
  (h : (a - b)^10 + (a - c)^10 = 1) : 
  |a - b| + |b - c| + |c - a| = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_differences_l2309_230929


namespace NUMINAMATH_CALUDE_t_equality_and_inequality_l2309_230905

/-- The function t(n, s) represents the maximum number of edges in a graph with n vertices
    that does not contain s independent vertices. -/
noncomputable def t (n s : ℕ) : ℕ := sorry

/-- Theorem stating the equality and inequality for t(n, s) -/
theorem t_equality_and_inequality (n s : ℕ) : 
  t (n - s) s + (n - s) * (s - 1) + (s.choose 2) = t n s ∧ 
  t n s ≤ ⌊((s - 1 : ℚ) / (2 * s : ℚ)) * (n^2 : ℚ)⌋ := by sorry

end NUMINAMATH_CALUDE_t_equality_and_inequality_l2309_230905


namespace NUMINAMATH_CALUDE_condition_analysis_l2309_230900

theorem condition_analysis (x y : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 0 → (x - 1) * (y - 2) = 0) ∧ 
  (∃ x y : ℝ, (x - 1) * (y - 2) = 0 ∧ (x - 1)^2 + (y - 2)^2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_analysis_l2309_230900


namespace NUMINAMATH_CALUDE_number_of_divisors_36_l2309_230927

theorem number_of_divisors_36 : Finset.card (Nat.divisors 36) = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_36_l2309_230927


namespace NUMINAMATH_CALUDE_total_odd_initial_plums_l2309_230916

def is_odd_initial (name : String) : Bool :=
  let initial := name.front
  let position := initial.toUpper.toNat - 'A'.toNat + 1
  position % 2 ≠ 0

def plums_picked (name : String) (amount : ℕ) : ℕ :=
  if is_odd_initial name then amount else 0

theorem total_odd_initial_plums :
  let melanie_plums := plums_picked "Melanie" 4
  let dan_plums := plums_picked "Dan" 9
  let sally_plums := plums_picked "Sally" 3
  let ben_plums := plums_picked "Ben" (2 * (4 + 9))
  let peter_plums := plums_picked "Peter" (((3 * 3) / 4) - ((3 * 1) / 4))
  melanie_plums + dan_plums + sally_plums + ben_plums + peter_plums = 7 :=
by sorry

end NUMINAMATH_CALUDE_total_odd_initial_plums_l2309_230916


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l2309_230993

/-- Given a line passing through points (1,3) and (3,7), 
    the sum of its slope and y-intercept is equal to 3. -/
theorem slope_intercept_sum (m b : ℝ) : 
  (3 = m * 1 + b) →   -- Line passes through (1,3)
  (7 = m * 3 + b) →   -- Line passes through (3,7)
  m + b = 3 :=
by sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_l2309_230993


namespace NUMINAMATH_CALUDE_l_shaped_area_is_23_l2309_230932

-- Define the side lengths
def large_square_side : ℝ := 8
def medium_square_side : ℝ := 4
def small_square_side : ℝ := 3

-- Define the areas
def large_square_area : ℝ := large_square_side ^ 2
def medium_square_area : ℝ := medium_square_side ^ 2
def small_square_area : ℝ := small_square_side ^ 2

-- Define the L-shaped area
def l_shaped_area : ℝ := large_square_area - (2 * medium_square_area + small_square_area)

-- Theorem statement
theorem l_shaped_area_is_23 : l_shaped_area = 23 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_area_is_23_l2309_230932


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l2309_230944

theorem deal_or_no_deal_probability (total_boxes : ℕ) (high_value_boxes : ℕ) (eliminated_boxes : ℕ) :
  total_boxes = 30 →
  high_value_boxes = 10 →
  eliminated_boxes = 20 →
  (total_boxes - eliminated_boxes : ℚ) / 2 ≤ high_value_boxes :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l2309_230944


namespace NUMINAMATH_CALUDE_angle_expression_value_l2309_230926

theorem angle_expression_value (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_value_l2309_230926


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_odd_divisible_by_three_l2309_230972

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

theorem sum_of_three_consecutive_odd_divisible_by_three : 
  ∀ (a b c : ℕ), 
    (is_odd a ∧ is_odd b ∧ is_odd c) → 
    (a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0) →
    (∃ k, b = a + 6*k + 6 ∧ c = b + 6) →
    (c = 27) →
    (a + b + c = 63) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_odd_divisible_by_three_l2309_230972


namespace NUMINAMATH_CALUDE_trigonometric_identity_proof_l2309_230970

theorem trigonometric_identity_proof (x : ℝ) : 
  Real.sin (x + Real.pi / 3) + 2 * Real.sin (x - Real.pi / 3) - Real.sqrt 3 * Real.cos (2 * Real.pi / 3 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_proof_l2309_230970


namespace NUMINAMATH_CALUDE_outfit_combinations_l2309_230942

theorem outfit_combinations (shirts : ℕ) (ties : ℕ) : shirts = 7 → ties = 6 → shirts * (ties + 1) = 49 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l2309_230942


namespace NUMINAMATH_CALUDE_equation_holds_l2309_230950

theorem equation_holds : (8 - 2) + 5 - (3 - 1) = 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l2309_230950


namespace NUMINAMATH_CALUDE_exactly_25_sixes_probability_l2309_230908

/-- A cube made of 27 dice -/
structure CubeOfDice :=
  (size : Nat)
  (h_size : size = 27)

/-- The probability of a specific outcome on the surface of the cube -/
def surface_probability (c : CubeOfDice) : ℚ :=
  31 / (2^13 * 3^18)

/-- Theorem: The probability of exactly 25 sixes on the surface of a cube made of 27 dice -/
theorem exactly_25_sixes_probability (c : CubeOfDice) : 
  surface_probability c = 31 / (2^13 * 3^18) := by
  sorry

end NUMINAMATH_CALUDE_exactly_25_sixes_probability_l2309_230908


namespace NUMINAMATH_CALUDE_rest_area_location_l2309_230983

/-- Represents a highway with exits and a rest area -/
structure Highway where
  fifth_exit : ℝ
  seventh_exit : ℝ
  rest_area : ℝ

/-- The rest area is located halfway between the fifth and seventh exits -/
def is_halfway (h : Highway) : Prop :=
  h.rest_area = (h.fifth_exit + h.seventh_exit) / 2

/-- Theorem: Given the conditions, prove that the rest area is at milepost 65 -/
theorem rest_area_location (h : Highway) 
    (h_fifth : h.fifth_exit = 35)
    (h_seventh : h.seventh_exit = 95)
    (h_halfway : is_halfway h) : 
    h.rest_area = 65 := by
  sorry

#check rest_area_location

end NUMINAMATH_CALUDE_rest_area_location_l2309_230983


namespace NUMINAMATH_CALUDE_train_length_l2309_230988

/-- Calculates the length of a train given its speed and time to cross an electric pole. -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) : 
  speed_kmh = 50.4 → time_sec = 20 → speed_kmh * (1000 / 3600) * time_sec = 280 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2309_230988


namespace NUMINAMATH_CALUDE_n_gon_determination_l2309_230915

/-- The number of elements required to determine an n-gon uniquely -/
def elementsRequired (n : ℕ) : ℕ := 2 * n - 3

/-- The minimum number of sides required among the elements -/
def minSidesRequired (n : ℕ) : ℕ := n - 2

/-- Predicate to check if a number is at least 3 -/
def isAtLeastThree (n : ℕ) : Prop := n ≥ 3

/-- Theorem stating the number of elements and minimum sides required to determine an n-gon -/
theorem n_gon_determination (n : ℕ) (h : isAtLeastThree n) :
  elementsRequired n = 2 * n - 3 ∧ minSidesRequired n = n - 2 :=
by sorry

end NUMINAMATH_CALUDE_n_gon_determination_l2309_230915


namespace NUMINAMATH_CALUDE_max_attendance_difference_l2309_230946

-- Define the estimates and error margins
def chloe_estimate : ℝ := 40000
def derek_estimate : ℝ := 55000
def emma_estimate : ℝ := 75000

def chloe_error : ℝ := 0.05
def derek_error : ℝ := 0.15
def emma_error : ℝ := 0.10

-- Define the ranges for actual attendances
def chicago_range : Set ℝ := {x | chloe_estimate * (1 - chloe_error) ≤ x ∧ x ≤ chloe_estimate * (1 + chloe_error)}
def denver_range : Set ℝ := {x | derek_estimate / (1 + derek_error) ≤ x ∧ x ≤ derek_estimate / (1 - derek_error)}
def miami_range : Set ℝ := {x | emma_estimate * (1 - emma_error) ≤ x ∧ x ≤ emma_estimate * (1 + emma_error)}

-- Define the theorem
theorem max_attendance_difference :
  ∃ (c d m : ℝ),
    c ∈ chicago_range ∧
    d ∈ denver_range ∧
    m ∈ miami_range ∧
    (⌊(max c (max d m) - min c (min d m) + 500) / 1000⌋ * 1000 = 45000) :=
sorry

end NUMINAMATH_CALUDE_max_attendance_difference_l2309_230946


namespace NUMINAMATH_CALUDE_product_of_polynomials_l2309_230966

theorem product_of_polynomials (d p q : ℝ) : 
  (4 * d^3 + 2 * d^2 - 5 * d + p) * (6 * d^2 + q * d - 3) = 
  24 * d^5 + q * d^4 - 33 * d^3 - 15 * d^2 + q * d - 15 → 
  p + q = 12.5 := by
sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l2309_230966


namespace NUMINAMATH_CALUDE_range_of_sum_equal_product_l2309_230918

theorem range_of_sum_equal_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = x * y) :
  x + y ≥ 4 ∧ ∀ z ≥ 4, ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = x * y ∧ x + y = z :=
by sorry

end NUMINAMATH_CALUDE_range_of_sum_equal_product_l2309_230918


namespace NUMINAMATH_CALUDE_parallel_vectors_l2309_230930

theorem parallel_vectors (a b : ℝ × ℝ) :
  a.1 = 2 ∧ a.2 = -1 ∧ b.2 = 3 ∧ a.1 * b.2 = a.2 * b.1 → b.1 = -6 :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_l2309_230930


namespace NUMINAMATH_CALUDE_polynomial_roots_magnitude_l2309_230933

theorem polynomial_roots_magnitude (c : ℂ) : 
  (∃ (Q : ℂ → ℂ), 
    Q = (fun x => (x^2 - 3*x + 3) * (x^2 - c*x + 9) * (x^2 - 5*x + 15)) ∧
    (∃ (r1 r2 r3 : ℂ), 
      r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧
      (∀ x : ℂ, Q x = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3))) →
  Complex.abs c = 6 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_magnitude_l2309_230933


namespace NUMINAMATH_CALUDE_vector_computation_l2309_230949

theorem vector_computation :
  let v1 : Fin 3 → ℝ := ![3, -5, 1]
  let v2 : Fin 3 → ℝ := ![-1, 4, -2]
  let v3 : Fin 3 → ℝ := ![2, -1, 3]
  2 • v1 + 3 • v2 - v3 = ![1, 3, -7] := by
  sorry

end NUMINAMATH_CALUDE_vector_computation_l2309_230949


namespace NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_lines_equation_l2309_230959

-- Define the slope of line l₁
def slope_l1 : ℚ := -3 / 4

-- Define a point that l₂ passes through
def point_l2 : ℚ × ℚ := (-1, 3)

-- Define the area of the triangle formed by l₂ and the coordinate axes
def triangle_area : ℚ := 4

-- Theorem for the parallel line
theorem parallel_line_equation :
  ∃ (c : ℚ), 3 * point_l2.1 + 4 * point_l2.2 + c = 0 ∧
  ∀ (x y : ℚ), 3 * x + 4 * y + c = 0 ↔ 3 * x + 4 * y - 9 = 0 :=
sorry

-- Theorem for the perpendicular lines
theorem perpendicular_lines_equation :
  ∃ (n : ℚ), (n^2 = 96) ∧
  (∀ (x y : ℚ), 4 * x - 3 * y + n = 0 ↔ 4 * x - 3 * y + 4 * Real.sqrt 6 = 0 ∨
                                        4 * x - 3 * y - 4 * Real.sqrt 6 = 0) ∧
  (1/2 * |n/4| * |n/3| = triangle_area) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_lines_equation_l2309_230959


namespace NUMINAMATH_CALUDE_triangle_altitude_length_l2309_230961

/-- Given a rectangle with sides a and b, and a triangle with its base as the diagonal of the rectangle
    and area twice that of the rectangle, the length of the altitude of the triangle to its base
    (the diagonal) is (4ab)/√(a² + b²). -/
theorem triangle_altitude_length (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let rectangle_area := a * b
  let diagonal := Real.sqrt (a^2 + b^2)
  let triangle_area := 2 * rectangle_area
  let altitude := (2 * triangle_area) / diagonal
  altitude = (4 * a * b) / Real.sqrt (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_length_l2309_230961


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2309_230982

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = 2 → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2309_230982


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l2309_230995

theorem least_k_for_inequality (k : ℤ) : 
  (∀ m : ℤ, m < k → (0.00010101 * (10 : ℝ)^m ≤ 100)) ∧ 
  (0.00010101 * (10 : ℝ)^k > 100) → 
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l2309_230995


namespace NUMINAMATH_CALUDE_exam_score_proof_l2309_230913

theorem exam_score_proof (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) :
  total_questions = 80 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 120 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 40 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_proof_l2309_230913


namespace NUMINAMATH_CALUDE_digit_difference_base_4_9_l2309_230911

theorem digit_difference_base_4_9 (n : ℕ) (h : n = 523) : 
  (Nat.log 4 n + 1) - (Nat.log 9 n + 1) = 2 := by sorry

end NUMINAMATH_CALUDE_digit_difference_base_4_9_l2309_230911


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_l2309_230986

theorem mark_and_carolyn_money : 
  (5 : ℚ) / 8 + (2 : ℚ) / 5 = (41 : ℚ) / 40 := by sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_l2309_230986


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_geq_2_l2309_230906

-- Define the function f(x) = |x-a|
def f (a : ℝ) (x : ℝ) : ℝ := abs (x - a)

-- State the theorem
theorem monotone_decreasing_implies_a_geq_2 (a : ℝ) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 2 → f a x ≥ f a y) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_geq_2_l2309_230906


namespace NUMINAMATH_CALUDE_rangers_apprentice_reading_l2309_230910

theorem rangers_apprentice_reading (total_books : Nat) (pages_per_book : Nat) 
  (books_read_first_month : Nat) (pages_left_to_finish : Nat) :
  total_books = 14 →
  pages_per_book = 200 →
  books_read_first_month = 4 →
  pages_left_to_finish = 1000 →
  (((total_books - books_read_first_month) * pages_per_book - pages_left_to_finish) / pages_per_book) / 
  (total_books - books_read_first_month) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rangers_apprentice_reading_l2309_230910


namespace NUMINAMATH_CALUDE_sin_negative_600_degrees_l2309_230904

theorem sin_negative_600_degrees : Real.sin (- 600 * π / 180) = - Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_600_degrees_l2309_230904


namespace NUMINAMATH_CALUDE_simplify_expression_l2309_230925

theorem simplify_expression (x y : ℝ) : (3 * x^2 * y^3)^2 = 9 * x^4 * y^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2309_230925


namespace NUMINAMATH_CALUDE_proposition_equivalence_l2309_230938

theorem proposition_equivalence (p q : Prop) : 
  (¬(p ∨ q)) → ((¬p) ∧ (¬q)) := by
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l2309_230938


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l2309_230991

theorem roots_sum_and_product (p q : ℝ) : 
  p^2 - 5*p + 6 = 0 → q^2 - 5*q + 6 = 0 → p^3 + p^4*q^2 + p^2*q^4 + q^3 = 503 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l2309_230991


namespace NUMINAMATH_CALUDE_break_even_price_l2309_230937

/-- Calculates the minimum selling price per component to break even -/
def minimum_selling_price (production_cost shipping_cost : ℚ) (fixed_costs : ℚ) (volume : ℕ) : ℚ :=
  (production_cost + shipping_cost + fixed_costs / volume)

theorem break_even_price 
  (production_cost : ℚ) 
  (shipping_cost : ℚ) 
  (fixed_costs : ℚ) 
  (volume : ℕ) 
  (h1 : production_cost = 80)
  (h2 : shipping_cost = 2)
  (h3 : fixed_costs = 16200)
  (h4 : volume = 150) :
  minimum_selling_price production_cost shipping_cost fixed_costs volume = 190 := by
  sorry

#eval minimum_selling_price 80 2 16200 150

end NUMINAMATH_CALUDE_break_even_price_l2309_230937


namespace NUMINAMATH_CALUDE_repair_cost_theorem_l2309_230936

def new_shoes_cost : ℝ := 28
def new_shoes_lifespan : ℝ := 2
def used_shoes_lifespan : ℝ := 1
def percentage_difference : ℝ := 0.2173913043478261

theorem repair_cost_theorem :
  ∃ (repair_cost : ℝ),
    repair_cost = 11.50 ∧
    (new_shoes_cost / new_shoes_lifespan) = repair_cost * (1 + percentage_difference) :=
by sorry

end NUMINAMATH_CALUDE_repair_cost_theorem_l2309_230936


namespace NUMINAMATH_CALUDE_projection_obtuse_implies_obtuse_projection_acute_inconclusive_l2309_230958

/-- Represents an angle --/
structure Angle where
  measure : ℝ
  is_positive : 0 < measure

/-- Represents the rectangular projection of an angle onto a plane --/
def rectangular_projection (α : Angle) : Angle :=
  sorry

/-- An angle is obtuse if its measure is greater than π/2 --/
def is_obtuse (α : Angle) : Prop :=
  α.measure > Real.pi / 2

/-- An angle is acute if its measure is less than π/2 --/
def is_acute (α : Angle) : Prop :=
  α.measure < Real.pi / 2

theorem projection_obtuse_implies_obtuse (α : Angle) :
  is_obtuse (rectangular_projection α) → is_obtuse α :=
sorry

theorem projection_acute_inconclusive (α : Angle) :
  is_acute (rectangular_projection α) → 
  (is_acute α ∨ is_obtuse α) :=
sorry

end NUMINAMATH_CALUDE_projection_obtuse_implies_obtuse_projection_acute_inconclusive_l2309_230958


namespace NUMINAMATH_CALUDE_toilet_paper_weeks_l2309_230955

/-- The number of bathrooms in the bed and breakfast -/
def num_bathrooms : ℕ := 6

/-- The number of rolls Stella stocks per bathroom per day -/
def rolls_per_bathroom_per_day : ℕ := 1

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of rolls in a pack (1 dozen) -/
def rolls_per_pack : ℕ := 12

/-- The number of packs Stella buys -/
def packs_bought : ℕ := 14

/-- The number of weeks Stella bought toilet paper for -/
def weeks_bought : ℚ :=
  (packs_bought * rolls_per_pack) / (num_bathrooms * rolls_per_bathroom_per_day * days_per_week)

theorem toilet_paper_weeks : weeks_bought = 4 := by sorry

end NUMINAMATH_CALUDE_toilet_paper_weeks_l2309_230955


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_l2309_230998

/-- The quadratic equation kx^2 + 18x + 2k = 0 has rational solutions if and only if k = 4, where k is a positive integer. -/
theorem quadratic_rational_solutions (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 18 * x + 2 * k = 0) ↔ k = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_l2309_230998


namespace NUMINAMATH_CALUDE_inverse_g_equals_five_l2309_230969

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^3 + 3

-- State the theorem
theorem inverse_g_equals_five (x : ℝ) : g (g⁻¹ x) = x → g⁻¹ x = 5 → x = 503 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_equals_five_l2309_230969


namespace NUMINAMATH_CALUDE_tax_calculation_correct_l2309_230957

/-- Calculates the personal income tax based on the given salary and tax brackets. -/
def calculate_tax (salary : ℕ) : ℕ :=
  let taxable_income := salary - 5000
  let first_bracket := min taxable_income 3000
  let second_bracket := min (taxable_income - 3000) 9000
  let third_bracket := max (taxable_income - 12000) 0
  (first_bracket * 3 + second_bracket * 10 + third_bracket * 20) / 100

/-- Theorem stating that the calculated tax for a salary of 20000 yuan is 1590 yuan. -/
theorem tax_calculation_correct :
  calculate_tax 20000 = 1590 := by sorry

end NUMINAMATH_CALUDE_tax_calculation_correct_l2309_230957


namespace NUMINAMATH_CALUDE_digit_sum_inequality_l2309_230975

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Condition for the existence of c_k -/
def HasValidCk (k : ℕ) : Prop :=
  ∃ (c_k : ℝ), c_k > 0 ∧ ∀ (n : ℕ), n > 0 → S (k * n) ≥ c_k * S n

/-- k has no prime divisors other than 2 or 5 -/
def HasOnly2And5Factors (k : ℕ) : Prop :=
  ∀ (p : ℕ), p.Prime → p ∣ k → p = 2 ∨ p = 5

/-- Main theorem -/
theorem digit_sum_inequality (k : ℕ) (h : k > 1) :
  HasValidCk k ↔ HasOnly2And5Factors k := by sorry

end NUMINAMATH_CALUDE_digit_sum_inequality_l2309_230975


namespace NUMINAMATH_CALUDE_sampling_survey_correct_l2309_230951

/-- Represents a statement about quality testing methods -/
inductive QualityTestingMethod
| SamplingSurvey
| Other

/-- Represents the correctness of a statement -/
inductive Correctness
| Correct
| Incorrect

/-- The correct method for testing the quality of a batch of light bulbs -/
def lightBulbQualityTestingMethod : QualityTestingMethod := QualityTestingMethod.SamplingSurvey

/-- Theorem stating that sampling survey is the correct method for testing light bulb quality -/
theorem sampling_survey_correct :
  Correctness.Correct = match lightBulbQualityTestingMethod with
    | QualityTestingMethod.SamplingSurvey => Correctness.Correct
    | QualityTestingMethod.Other => Correctness.Incorrect :=
by sorry

end NUMINAMATH_CALUDE_sampling_survey_correct_l2309_230951


namespace NUMINAMATH_CALUDE_total_profit_is_100_l2309_230965

/-- Calculates the total profit given investments, time periods, and A's share --/
def calculate_total_profit (a_investment : ℕ) (a_months : ℕ) (b_investment : ℕ) (b_months : ℕ) (a_share : ℕ) : ℕ :=
  let a_weight := a_investment * a_months
  let b_weight := b_investment * b_months
  let total_weight := a_weight + b_weight
  let part_value := a_share * total_weight / a_weight
  part_value

theorem total_profit_is_100 :
  calculate_total_profit 150 12 200 6 60 = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_100_l2309_230965


namespace NUMINAMATH_CALUDE_complex_number_opposite_parts_l2309_230917

theorem complex_number_opposite_parts (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_opposite_parts_l2309_230917


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2309_230924

theorem min_value_quadratic (b : ℝ) : 
  (∀ x : ℝ, x^2 - 12*x + 32 ≤ 0 → b ≤ x) → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2309_230924


namespace NUMINAMATH_CALUDE_stephanie_remaining_payment_l2309_230953

/-- Represents the bills and payments in Stephanie's household budget --/
structure BudgetInfo where
  electricity_bill : ℝ
  gas_bill : ℝ
  water_bill : ℝ
  internet_bill : ℝ
  gas_initial_payment_fraction : ℝ
  gas_additional_payment : ℝ
  water_payment_fraction : ℝ
  internet_payment_count : ℕ
  internet_payment_amount : ℝ

/-- Calculates the remaining amount to pay given the budget information --/
def remaining_payment (budget : BudgetInfo) : ℝ :=
  let total_bills := budget.electricity_bill + budget.gas_bill + budget.water_bill + budget.internet_bill
  let total_paid := budget.electricity_bill +
                    (budget.gas_bill * budget.gas_initial_payment_fraction + budget.gas_additional_payment) +
                    (budget.water_bill * budget.water_payment_fraction) +
                    (budget.internet_payment_count : ℝ) * budget.internet_payment_amount
  total_bills - total_paid

/-- Theorem stating that the remaining payment for Stephanie's bills is $30 --/
theorem stephanie_remaining_payment :
  let budget : BudgetInfo := {
    electricity_bill := 60,
    gas_bill := 40,
    water_bill := 40,
    internet_bill := 25,
    gas_initial_payment_fraction := 0.75,
    gas_additional_payment := 5,
    water_payment_fraction := 0.5,
    internet_payment_count := 4,
    internet_payment_amount := 5
  }
  remaining_payment budget = 30 := by sorry

end NUMINAMATH_CALUDE_stephanie_remaining_payment_l2309_230953


namespace NUMINAMATH_CALUDE_fraction_equality_l2309_230922

theorem fraction_equality (x : ℝ) : 
  (3/4 : ℝ) * (1/2 : ℝ) * (2/5 : ℝ) * x = 750.0000000000001 → x = 5000.000000000001 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2309_230922


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2309_230999

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 5 + a 6 + a 7 + a 8 = 20) →
  (a 1 + a 12 = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2309_230999


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_concentric_circles_l2309_230976

-- Define the structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a circle with center and radius
structure Circle where
  center : Point2D
  radius : ℝ

-- Define an equilateral triangle
structure EquilateralTriangle where
  a : Point2D
  b : Point2D
  c : Point2D

-- Function to check if a point lies on a circle
def pointOnCircle (p : Point2D) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Theorem statement
theorem equilateral_triangle_on_concentric_circles 
  (center : Point2D) (r₁ r₂ r₃ : ℝ) 
  (h₁ : 0 < r₁) (h₂ : r₁ < r₂) (h₃ : r₂ < r₃) :
  ∃ (t : EquilateralTriangle),
    pointOnCircle t.a (Circle.mk center r₂) ∧
    pointOnCircle t.b (Circle.mk center r₁) ∧
    pointOnCircle t.c (Circle.mk center r₃) :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_on_concentric_circles_l2309_230976


namespace NUMINAMATH_CALUDE_four_solutions_implies_a_greater_than_two_l2309_230985

-- Define the equation
def equation (a x : ℝ) : Prop := |x^3 - a*x^2| = x

-- Theorem statement
theorem four_solutions_implies_a_greater_than_two (a : ℝ) :
  (∃ w x y z : ℝ, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    equation a w ∧ equation a x ∧ equation a y ∧ equation a z) →
  a > 2 := by
  sorry

end NUMINAMATH_CALUDE_four_solutions_implies_a_greater_than_two_l2309_230985


namespace NUMINAMATH_CALUDE_min_records_theorem_l2309_230990

/-- The number of different labels -/
def n : ℕ := 50

/-- The total number of records -/
def total_records : ℕ := n * (n + 1) / 2

/-- The number of records we want to ensure have the same label -/
def target : ℕ := 10

/-- The function that calculates the minimum number of records to draw -/
def min_records_to_draw : ℕ := 
  (target - 1) * (n - (target - 1)) + (target - 1) * target / 2

/-- Theorem stating the minimum number of records to draw -/
theorem min_records_theorem : 
  min_records_to_draw = 415 := by sorry

end NUMINAMATH_CALUDE_min_records_theorem_l2309_230990


namespace NUMINAMATH_CALUDE_product_comparison_l2309_230909

theorem product_comparison (a : Fin 10 → ℝ) 
  (h_pos : ∀ i, a i > 0) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) : 
  (∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (∀ l m, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ m ≠ i ∧ m ≠ j ∧ m ≠ k ∧ l ≠ m → 
      a i * a j * a k > a l * a m)) ∨
  (∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (∀ l m n o, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ m ≠ i ∧ m ≠ j ∧ m ≠ k ∧ 
      n ≠ i ∧ n ≠ j ∧ n ≠ k ∧ o ≠ i ∧ o ≠ j ∧ o ≠ k ∧ 
      l ≠ m ∧ l ≠ n ∧ l ≠ o ∧ m ≠ n ∧ m ≠ o ∧ n ≠ o → 
      a i * a j * a k > a l * a m * a n * a o)) := by
sorry

end NUMINAMATH_CALUDE_product_comparison_l2309_230909


namespace NUMINAMATH_CALUDE_business_school_size_l2309_230971

/-- The number of students in the law school -/
def law_students : ℕ := 800

/-- The number of sibling pairs -/
def sibling_pairs : ℕ := 30

/-- The probability of selecting a sibling pair -/
def sibling_pair_probability : ℚ := 75 / 1000000

/-- The number of students in the business school -/
def business_students : ℕ := 5000

theorem business_school_size :
  (sibling_pairs : ℚ) / (business_students * law_students) = sibling_pair_probability :=
by sorry

end NUMINAMATH_CALUDE_business_school_size_l2309_230971


namespace NUMINAMATH_CALUDE_cos_B_value_triangle_area_l2309_230919

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the specific conditions for our triangle
def SpecialTriangle (t : Triangle) : Prop :=
  t.b = t.c ∧ 2 * Real.sin t.B = Real.sqrt 3 * Real.sin t.A

-- Theorem for part (i)
theorem cos_B_value (t : Triangle) (h : SpecialTriangle t) : 
  Real.cos t.B = Real.sqrt 3 / 3 := by
  sorry

-- Theorem for part (ii)
theorem triangle_area (t : Triangle) (h : SpecialTriangle t) (ha : t.a = 2) :
  (1/2) * t.a * t.b * Real.sin t.B = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_B_value_triangle_area_l2309_230919


namespace NUMINAMATH_CALUDE_base6_addition_l2309_230984

/-- Converts a base 6 number represented as a list of digits to its decimal (base 10) equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Converts a decimal (base 10) number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The main theorem stating that 3454₆ + 12345₆ = 142042₆ in base 6 -/
theorem base6_addition :
  decimalToBase6 (base6ToDecimal [3, 4, 5, 4] + base6ToDecimal [1, 2, 3, 4, 5]) =
  [1, 4, 2, 0, 4, 2] := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_l2309_230984


namespace NUMINAMATH_CALUDE_p_iff_q_l2309_230921

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x - y - 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a*y - 2 = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, f x₁ y₁ ∧ f x₂ y₂ ∧ g x₁ y₁ ∧ g x₂ y₂ →
    (y₂ - y₁) / (x₂ - x₁) = (y₂ - y₁) / (x₂ - x₁)

-- Define the propositions p and q
def p (a : ℝ) : Prop := parallel (l₁) (l₂ a)
def q (a : ℝ) : Prop := a = -1

-- State the theorem
theorem p_iff_q : ∀ a : ℝ, p a ↔ q a := by sorry

end NUMINAMATH_CALUDE_p_iff_q_l2309_230921


namespace NUMINAMATH_CALUDE_max_reverse_digit_diff_l2309_230912

/-- Given two two-digit positive integers with the same digits in reverse order and
    their positive difference less than 60, the maximum difference is 54 -/
theorem max_reverse_digit_diff :
  ∀ q r : ℕ,
  10 ≤ q ∧ q < 100 →  -- q is a two-digit number
  10 ≤ r ∧ r < 100 →  -- r is a two-digit number
  ∃ a b : ℕ,
    0 ≤ a ∧ a ≤ 9 ∧   -- a is a digit
    0 ≤ b ∧ b ≤ 9 ∧   -- b is a digit
    q = 10 * a + b ∧  -- q's representation
    r = 10 * b + a ∧  -- r's representation
    (q > r → q - r < 60) ∧  -- positive difference less than 60
    (r > q → r - q < 60) →
  (∀ q' r' : ℕ,
    (∃ a' b' : ℕ,
      0 ≤ a' ∧ a' ≤ 9 ∧
      0 ≤ b' ∧ b' ≤ 9 ∧
      q' = 10 * a' + b' ∧
      r' = 10 * b' + a' ∧
      (q' > r' → q' - r' < 60) ∧
      (r' > q' → r' - q' < 60)) →
    q' - r' ≤ 54) ∧
  ∃ q₀ r₀ : ℕ, q₀ - r₀ = 54 ∧
    (∃ a₀ b₀ : ℕ,
      0 ≤ a₀ ∧ a₀ ≤ 9 ∧
      0 ≤ b₀ ∧ b₀ ≤ 9 ∧
      q₀ = 10 * a₀ + b₀ ∧
      r₀ = 10 * b₀ + a₀ ∧
      q₀ - r₀ < 60) :=
by sorry

end NUMINAMATH_CALUDE_max_reverse_digit_diff_l2309_230912


namespace NUMINAMATH_CALUDE_total_area_is_62_l2309_230945

/-- The area of a figure composed of three rectangles -/
def figure_area (area1 area2 area3 : ℕ) : ℕ := area1 + area2 + area3

/-- Theorem: The total area of the figure is 62 square units -/
theorem total_area_is_62 (area1 area2 area3 : ℕ) 
  (h1 : area1 = 30) 
  (h2 : area2 = 12) 
  (h3 : area3 = 20) : 
  figure_area area1 area2 area3 = 62 := by
  sorry

#eval figure_area 30 12 20

end NUMINAMATH_CALUDE_total_area_is_62_l2309_230945


namespace NUMINAMATH_CALUDE_vector_dot_product_sum_l2309_230928

theorem vector_dot_product_sum (a b : ℝ × ℝ) : 
  a = (1/2, Real.sqrt 3/2) → 
  b = (-Real.sqrt 3/2, 1/2) → 
  (a.1 + b.1, a.2 + b.2) • a = 1 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_sum_l2309_230928


namespace NUMINAMATH_CALUDE_binomial_probability_three_out_of_six_l2309_230987

/-- The probability mass function for a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The problem statement -/
theorem binomial_probability_three_out_of_six :
  binomial_pmf 6 (1/2) 3 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_three_out_of_six_l2309_230987


namespace NUMINAMATH_CALUDE_opponent_total_score_l2309_230940

theorem opponent_total_score (team_scores : List Nat) 
  (h1 : team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  (h2 : ∃ lost_games : List Nat, lost_games.length = 6 ∧ 
    ∀ score ∈ lost_games, score + 1 ∈ team_scores)
  (h3 : ∃ double_score_games : List Nat, double_score_games.length = 5 ∧ 
    ∀ score ∈ double_score_games, 2 * (score / 2) ∈ team_scores)
  (h4 : ∃ tie_score : Nat, tie_score ∈ team_scores ∧ 
    tie_score ∉ (Classical.choose h2) ∧ tie_score ∉ (Classical.choose h3)) :
  (team_scores.sum + 6 - (Classical.choose h3).sum / 2 - (Classical.choose h4)) = 69 := by
sorry

end NUMINAMATH_CALUDE_opponent_total_score_l2309_230940


namespace NUMINAMATH_CALUDE_new_species_growth_pattern_l2309_230902

/-- Represents the shape of population growth --/
inductive GrowthShape
  | J
  | S

/-- Represents a species in a new area --/
structure Species where
  isNew : Bool
  populationSize : ℕ → ℕ  -- population size as a function of time
  growthPattern : List GrowthShape
  kValue : ℕ

/-- The maximum population allowed by environmental conditions --/
def environmentalCapacity (s : Species) : ℕ := s.kValue

theorem new_species_growth_pattern (s : Species) 
  (h1 : s.isNew = true) 
  (h2 : ∀ t, s.populationSize (t + 1) ≠ s.populationSize t) 
  (h3 : s.growthPattern.length ≥ 2) 
  (h4 : ∃ t, ∀ t' ≥ t, s.populationSize t' = s.kValue) 
  (h5 : s.kValue = environmentalCapacity s) :
  s.growthPattern = [GrowthShape.J, GrowthShape.S] := by
  sorry

end NUMINAMATH_CALUDE_new_species_growth_pattern_l2309_230902


namespace NUMINAMATH_CALUDE_age_difference_proof_l2309_230981

theorem age_difference_proof (younger_age elder_age : ℕ) 
  (h1 : younger_age = 30)
  (h2 : elder_age = 50)
  (h3 : elder_age - 5 = 5 * (younger_age - 5)) :
  elder_age - younger_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2309_230981


namespace NUMINAMATH_CALUDE_ratio_problem_l2309_230964

theorem ratio_problem (a b x m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 4 / 5) 
  (h4 : x = a + 0.25 * a) (h5 : m = b - 0.6 * b) : m / x = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2309_230964


namespace NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_2_l2309_230960

theorem factorization_of_2m_squared_minus_2 (m : ℝ) : 2 * m^2 - 2 = 2 * (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_2_l2309_230960


namespace NUMINAMATH_CALUDE_triangle_inequality_l2309_230974

theorem triangle_inequality (a b c r s : ℝ) :
  a > 0 → b > 0 → c > 0 → r > 0 →
  s = (a + b + c) / 2 →
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  1 / (s - a)^2 + 1 / (s - b)^2 + 1 / (s - c)^2 ≥ 1 / r^2 := by
sorry


end NUMINAMATH_CALUDE_triangle_inequality_l2309_230974


namespace NUMINAMATH_CALUDE_lacsap_hospital_staff_product_l2309_230994

/-- Represents the Lacsap Hospital staff composition -/
structure HospitalStaff where
  doctors_excluding_emily : ℕ
  nurses_excluding_robert : ℕ
  emily_is_doctor : Bool
  robert_is_nurse : Bool

/-- Calculates the total number of doctors -/
def total_doctors (staff : HospitalStaff) : ℕ :=
  staff.doctors_excluding_emily + (if staff.emily_is_doctor then 1 else 0)

/-- Calculates the total number of nurses -/
def total_nurses (staff : HospitalStaff) : ℕ :=
  staff.nurses_excluding_robert + (if staff.robert_is_nurse then 1 else 0)

/-- Calculates the number of doctors excluding Robert -/
def doctors_excluding_robert (staff : HospitalStaff) : ℕ :=
  total_doctors staff

/-- Calculates the number of nurses excluding Robert -/
def nurses_excluding_robert (staff : HospitalStaff) : ℕ :=
  staff.nurses_excluding_robert

theorem lacsap_hospital_staff_product :
  ∀ (staff : HospitalStaff),
    staff.doctors_excluding_emily = 5 →
    staff.nurses_excluding_robert = 3 →
    staff.emily_is_doctor = true →
    staff.robert_is_nurse = true →
    (doctors_excluding_robert staff) * (nurses_excluding_robert staff) = 12 := by
  sorry

end NUMINAMATH_CALUDE_lacsap_hospital_staff_product_l2309_230994


namespace NUMINAMATH_CALUDE_basketball_shots_l2309_230923

theorem basketball_shots (shots_made : ℝ) (shots_missed : ℝ) :
  shots_made = 0.8 * (shots_made + shots_missed) →
  shots_missed = 4 →
  shots_made + shots_missed = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_shots_l2309_230923


namespace NUMINAMATH_CALUDE_round_robin_equation_l2309_230996

/-- Represents a round-robin tournament -/
structure RoundRobinTournament where
  teams : ℕ
  total_games : ℕ
  games_formula : total_games = teams * (teams - 1) / 2

/-- Theorem: In a round-robin tournament with 45 total games, the equation x(x-1) = 2 * 45 holds true -/
theorem round_robin_equation (t : RoundRobinTournament) (h : t.total_games = 45) :
  t.teams * (t.teams - 1) = 2 * 45 := by
  sorry


end NUMINAMATH_CALUDE_round_robin_equation_l2309_230996


namespace NUMINAMATH_CALUDE_fraction_equality_problem_l2309_230967

theorem fraction_equality_problem (y : ℝ) : 
  (4 + y) / (6 + y) = (2 + y) / (3 + y) ↔ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_problem_l2309_230967


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2309_230977

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x = 1) → ¬(x^2 = 1)) ↔ (¬(x = 1) → (x ≠ 1 ∧ x ≠ -1)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2309_230977
