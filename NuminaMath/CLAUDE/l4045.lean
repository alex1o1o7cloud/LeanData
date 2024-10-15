import Mathlib

namespace NUMINAMATH_CALUDE_single_burger_cost_l4045_404598

theorem single_burger_cost 
  (total_spent : ℚ)
  (total_hamburgers : ℕ)
  (double_burgers : ℕ)
  (double_burger_cost : ℚ)
  (h1 : total_spent = 64.5)
  (h2 : total_hamburgers = 50)
  (h3 : double_burgers = 29)
  (h4 : double_burger_cost = 1.5)
  : ∃ (single_burger_cost : ℚ), single_burger_cost = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_single_burger_cost_l4045_404598


namespace NUMINAMATH_CALUDE_earthworm_catches_centipede_l4045_404502

/-- The time it takes for an earthworm to catch up to a centipede under specific conditions -/
theorem earthworm_catches_centipede : 
  let centipede_speed : ℚ := 5 / 3  -- meters per minute
  let earthworm_speed : ℚ := 5 / 2  -- meters per minute
  let initial_distance : ℚ := 20   -- meters
  let relative_speed : ℚ := earthworm_speed - centipede_speed
  let catch_up_time : ℚ := initial_distance / relative_speed
  catch_up_time = 24 := by sorry

end NUMINAMATH_CALUDE_earthworm_catches_centipede_l4045_404502


namespace NUMINAMATH_CALUDE_skew_lines_parallel_implication_l4045_404576

/-- Two lines in 3D space -/
structure Line3D where
  -- This is a simplified representation of a line in 3D space
  -- In a real implementation, we might use vectors or points to define a line

/-- Predicate for two lines being skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Two lines are skew if they are not parallel and do not intersect
  sorry

/-- Predicate for two lines being parallel -/
def are_parallel (l1 l2 : Line3D) : Prop :=
  -- Two lines are parallel if they have the same direction but do not intersect
  sorry

theorem skew_lines_parallel_implication (a b c : Line3D) 
  (h1 : are_skew a b) (h2 : are_parallel c a) : 
  ¬(are_parallel c b) := by
  sorry

end NUMINAMATH_CALUDE_skew_lines_parallel_implication_l4045_404576


namespace NUMINAMATH_CALUDE_f_lower_bound_and_g_inequality_l4045_404594

noncomputable section

def f (x : ℝ) := x - Real.log x

def g (x : ℝ) := x^3 + x^2 * (f x) - 16*x

theorem f_lower_bound_and_g_inequality {x : ℝ} (hx : x > 0) :
  f x ≥ 1 ∧ g x > -20 := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_and_g_inequality_l4045_404594


namespace NUMINAMATH_CALUDE_exam_average_marks_l4045_404585

theorem exam_average_marks (total_boys : ℕ) (passed_boys : ℕ) (all_average : ℚ) (passed_average : ℚ) 
  (h1 : total_boys = 120)
  (h2 : passed_boys = 110)
  (h3 : all_average = 37)
  (h4 : passed_average = 39) :
  let failed_boys := total_boys - passed_boys
  let total_marks := total_boys * all_average
  let passed_marks := passed_boys * passed_average
  let failed_marks := total_marks - passed_marks
  failed_marks / failed_boys = 15 := by
sorry

end NUMINAMATH_CALUDE_exam_average_marks_l4045_404585


namespace NUMINAMATH_CALUDE_distance_from_movements_l4045_404536

/-- The distance between two points given their net movements --/
theorem distance_from_movements (south west : ℝ) (south_nonneg : 0 ≤ south) (west_nonneg : 0 ≤ west) :
  Real.sqrt (south ^ 2 + west ^ 2) = 50 ↔ south = 30 ∧ west = 40 := by
sorry

end NUMINAMATH_CALUDE_distance_from_movements_l4045_404536


namespace NUMINAMATH_CALUDE_olympic_medal_distribution_count_l4045_404561

/-- Represents the number of sprinters of each nationality -/
structure SprinterCounts where
  total : Nat
  americans : Nat
  kenyans : Nat
  others : Nat

/-- Represents the constraints on medal distribution -/
structure MedalConstraints where
  max_american_medals : Nat
  min_kenyan_medals : Nat

/-- Calculates the number of ways to award medals given the sprinter counts and constraints -/
def count_medal_distributions (counts : SprinterCounts) (constraints : MedalConstraints) : Nat :=
  sorry

/-- Theorem stating the number of ways to award medals in the given scenario -/
theorem olympic_medal_distribution_count :
  let counts : SprinterCounts := ⟨10, 4, 2, 4⟩
  let constraints : MedalConstraints := ⟨1, 1⟩
  count_medal_distributions counts constraints = 360 := by
  sorry

end NUMINAMATH_CALUDE_olympic_medal_distribution_count_l4045_404561


namespace NUMINAMATH_CALUDE_mary_peter_ratio_l4045_404520

/-- Represents the amount of chestnuts picked by each person in kilograms -/
structure ChestnutPickers where
  mary : ℝ
  peter : ℝ
  lucy : ℝ

/-- The conditions of the chestnut picking problem -/
def chestnut_problem (c : ChestnutPickers) : Prop :=
  c.mary = 12 ∧
  c.lucy = c.peter + 2 ∧
  c.mary + c.peter + c.lucy = 26

/-- The theorem stating the ratio of Mary's chestnuts to Peter's is 2:1 -/
theorem mary_peter_ratio (c : ChestnutPickers) :
  chestnut_problem c → c.mary / c.peter = 2 := by
  sorry

#check mary_peter_ratio

end NUMINAMATH_CALUDE_mary_peter_ratio_l4045_404520


namespace NUMINAMATH_CALUDE_initial_apples_count_apple_problem_l4045_404564

theorem initial_apples_count (num_trees : ℕ) (apples_per_tree : ℕ) (remaining_apples : ℕ) : ℕ :=
  num_trees * apples_per_tree + remaining_apples

theorem apple_problem : initial_apples_count 3 8 9 = 33 := by
  sorry

end NUMINAMATH_CALUDE_initial_apples_count_apple_problem_l4045_404564


namespace NUMINAMATH_CALUDE_y_intercept_comparison_l4045_404546

def f (x : ℝ) := x^2 - 2*x + 5
def g (x : ℝ) := x^2 + 2*x + 3

theorem y_intercept_comparison : f 0 > g 0 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_comparison_l4045_404546


namespace NUMINAMATH_CALUDE_root_of_equation_l4045_404539

theorem root_of_equation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x > 0, (Real.sqrt (a * b * x * (a + b + x)) + 
             Real.sqrt (b * c * x * (b + c + x)) + 
             Real.sqrt (c * a * x * (c + a + x)) = 
             Real.sqrt (a * b * c * (a + b + c))) ∧
           (x = (a * b * c) / (a * b + b * c + c * a + 2 * Real.sqrt (a * b * c * (a + b + c)))) :=
by sorry

end NUMINAMATH_CALUDE_root_of_equation_l4045_404539


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4045_404597

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4045_404597


namespace NUMINAMATH_CALUDE_sqrt_less_than_3x_plus_1_l4045_404538

theorem sqrt_less_than_3x_plus_1 (x : ℝ) (hx : x > 0) : Real.sqrt x < 3 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_less_than_3x_plus_1_l4045_404538


namespace NUMINAMATH_CALUDE_polynomial_factor_theorem_l4045_404589

theorem polynomial_factor_theorem (a b c : ℤ) :
  (∃ d e : ℤ, (X^3 - X^2 - X - 1) * (d * X + e) = a * X^4 + b * X^3 + c * X^2 + 1) →
  c = 1 - a :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factor_theorem_l4045_404589


namespace NUMINAMATH_CALUDE_sqrt_simplification_l4045_404593

theorem sqrt_simplification : 
  Real.sqrt 45 - 2 * Real.sqrt 5 + Real.sqrt 360 / Real.sqrt 2 = Real.sqrt 245 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l4045_404593


namespace NUMINAMATH_CALUDE_equation_satisfied_l4045_404521

theorem equation_satisfied (x y : ℝ) (hx : x = 2) (hy : y = -1) :
  3 * x - 4 * y = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l4045_404521


namespace NUMINAMATH_CALUDE_vasya_wins_l4045_404512

/-- Represents a game state with a list of piles of stones -/
structure GameState where
  piles : List Nat

/-- Represents a player in the game -/
inductive Player
  | Petya
  | Vasya

/-- Defines a valid move in the game -/
def validMove (gs : GameState) (gs' : GameState) : Prop :=
  ∃ (i j : Nat) (a b c d : Nat),
    i < gs.piles.length ∧
    j < gs.piles.length ∧
    i ≠ j ∧
    a + b + c + d = gs.piles[i]! + gs.piles[j]! ∧
    gs'.piles = (gs.piles.removeNth i).removeNth j ++ [a, b, c, d]

/-- Defines the initial game state -/
def initialState : GameState :=
  { piles := [40, 40, 40] }

/-- Determines if a game state is terminal (no more moves possible) -/
def isTerminal (gs : GameState) : Prop :=
  ∀ gs', ¬validMove gs gs'

/-- Theorem stating that Vasya has a winning strategy -/
theorem vasya_wins :
  ∃ (strategy : GameState → GameState),
    (∀ gs gs', validMove gs gs' → 
      validMove (strategy gs') (strategy (strategy gs'))) →
    (∀ gs, isTerminal (strategy gs) → 
      ∃ n, n % 2 = 0 ∧ 
        (initialState = gs ∨ 
         ∃ gs₁ gs₂, validMove gs₁ gs₂ ∧ 
           (gs₂ = gs ∨ strategy gs₂ = gs))) :=
  sorry


end NUMINAMATH_CALUDE_vasya_wins_l4045_404512


namespace NUMINAMATH_CALUDE_min_coins_needed_l4045_404554

/-- The cost of the sneakers in cents -/
def cost : ℕ := 4550

/-- The amount Chloe already has in cents (four $10 bills and ten quarters) -/
def existing_funds : ℕ := 4250

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The minimum number of additional coins (dimes and nickels) needed -/
def min_coins : ℕ := 30

theorem min_coins_needed :
  ∀ (d n : ℕ),
    d * dime_value + n * nickel_value + existing_funds ≥ cost →
    d + n ≥ min_coins :=
sorry

end NUMINAMATH_CALUDE_min_coins_needed_l4045_404554


namespace NUMINAMATH_CALUDE_gift_card_value_l4045_404578

/-- Represents the gift card problem -/
theorem gift_card_value (cost_per_pound : ℝ) (pounds_bought : ℝ) (remaining_balance : ℝ) :
  cost_per_pound = 8.58 →
  pounds_bought = 4.0 →
  remaining_balance = 35.68 →
  cost_per_pound * pounds_bought + remaining_balance = 70.00 := by
  sorry


end NUMINAMATH_CALUDE_gift_card_value_l4045_404578


namespace NUMINAMATH_CALUDE_distribute_and_simplify_l4045_404569

theorem distribute_and_simplify (a : ℝ) :
  (-12 * a) * (2 * a^2 - 2/3 * a + 5/6) = -24 * a^3 + 8 * a^2 - 10 * a := by
  sorry

end NUMINAMATH_CALUDE_distribute_and_simplify_l4045_404569


namespace NUMINAMATH_CALUDE_paint_for_sun_l4045_404542

/-- The amount of paint left for the sun, given Mary's and Mike's usage --/
def paint_left_for_sun (mary_paint : ℝ) (mike_extra_paint : ℝ) (total_paint : ℝ) : ℝ :=
  total_paint - (mary_paint + (mary_paint + mike_extra_paint))

/-- Theorem stating the amount of paint left for the sun --/
theorem paint_for_sun :
  paint_left_for_sun 3 2 13 = 5 := by
  sorry

end NUMINAMATH_CALUDE_paint_for_sun_l4045_404542


namespace NUMINAMATH_CALUDE_inverse_implies_odd_l4045_404513

-- Define the function f on the real numbers
variable (f : ℝ → ℝ)

-- Define the property of f being invertible
def IsInvertible (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Define the inverse function property given in the problem
def InverseFunctionProperty (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, IsInvertible f ∧ (∀ x, g (-x) = f⁻¹ (-x))

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- The theorem to prove
theorem inverse_implies_odd (f : ℝ → ℝ) :
  InverseFunctionProperty f → IsOdd f :=
sorry

end NUMINAMATH_CALUDE_inverse_implies_odd_l4045_404513


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l4045_404555

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- Theorem stating the total distance traveled by the ball -/
theorem ball_bounce_distance :
  let initialHeight : ℝ := 80
  let reboundFactor : ℝ := 3/4
  let bounces : ℕ := 4
  totalDistance initialHeight reboundFactor bounces = 357.5 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l4045_404555


namespace NUMINAMATH_CALUDE_march_production_3000_l4045_404580

/-- Represents the number of months since March -/
def months_since_march : Nat → Nat
  | 0 => 0  -- March
  | 1 => 1  -- April
  | 2 => 2  -- May
  | 3 => 3  -- June
  | 4 => 4  -- July
  | n + 5 => n + 5

/-- Calculates the mask production for a given month based on the initial production in March -/
def mask_production (initial_production : Nat) (month : Nat) : Nat :=
  initial_production * (2 ^ (months_since_march month))

theorem march_production_3000 :
  ∃ (initial_production : Nat),
    mask_production initial_production 4 = 48000 ∧
    initial_production = 3000 := by
  sorry

end NUMINAMATH_CALUDE_march_production_3000_l4045_404580


namespace NUMINAMATH_CALUDE_game_points_l4045_404515

/-- Given a game where two players earn 10 points for winning a round,
    if they play 8 matches and one player wins 3/4 of the matches,
    prove that the other player earns 20 points. -/
theorem game_points (total_matches : ℕ) (points_per_win : ℕ) 
  (winner_fraction : ℚ) (h1 : total_matches = 8) 
  (h2 : points_per_win = 10) (h3 : winner_fraction = 3/4) : 
  (total_matches - (winner_fraction * total_matches).num) * points_per_win = 20 :=
by sorry

end NUMINAMATH_CALUDE_game_points_l4045_404515


namespace NUMINAMATH_CALUDE_cubic_root_cube_relation_l4045_404553

/-- Given a cubic polynomial f(x) = x^3 + 2x^2 + 3x + 4, there exists another cubic polynomial 
g(x) = x^3 + bx^2 + cx + d such that the roots of g(x) are the cubes of the roots of f(x). -/
theorem cubic_root_cube_relation : 
  ∃ (b c d : ℝ), ∀ (r : ℂ), (r^3 + 2*r^2 + 3*r + 4 = 0) → 
    ((r^3)^3 + b*(r^3)^2 + c*(r^3) + d = 0) := by
  sorry


end NUMINAMATH_CALUDE_cubic_root_cube_relation_l4045_404553


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l4045_404518

/-- Proves that a bus stopping for 20 minutes per hour with an average speed of 36 kmph including stoppages has a speed of 54 kmph excluding stoppages. -/
theorem bus_speed_excluding_stoppages 
  (average_speed : ℝ) 
  (stopping_time : ℝ) 
  (h1 : average_speed = 36) 
  (h2 : stopping_time = 20) : ℝ := by
  sorry

#check bus_speed_excluding_stoppages

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l4045_404518


namespace NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l4045_404551

theorem other_root_of_complex_quadratic (z : ℂ) :
  z^2 = -91 + 84*I ∧ z = 7 + 12*I → (-z) = -7 - 12*I ∧ (-z)^2 = -91 + 84*I := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l4045_404551


namespace NUMINAMATH_CALUDE_parabola_focus_l4045_404560

/-- A parabola is defined by the equation y = 8x^2 -/
def parabola_equation (x y : ℝ) : Prop := y = 8 * x^2

/-- The focus of a parabola is a point on its axis of symmetry -/
def is_focus (x y : ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∃ (p : ℝ), p > 0 ∧ parabola x y ∧
  ∀ (x' y' : ℝ), parabola x' y' → (x' - x)^2 + (y' - y)^2 = (y' + y - 4*p)^2 / 4

/-- The focus of the parabola y = 8x^2 has coordinates (0, 1/32) -/
theorem parabola_focus :
  is_focus 0 (1/32) parabola_equation :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l4045_404560


namespace NUMINAMATH_CALUDE_sqrt_meaningful_l4045_404574

theorem sqrt_meaningful (x : ℝ) : (∃ y : ℝ, y^2 = x - 2) → x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_l4045_404574


namespace NUMINAMATH_CALUDE_larger_number_problem_l4045_404588

theorem larger_number_problem (x y : ℝ) : 
  x > y → x + y = 40 → x * y = 96 → x = 37.435 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l4045_404588


namespace NUMINAMATH_CALUDE_solve_equation_l4045_404526

theorem solve_equation (x : ℝ) (h : 9 / (1 + 4 / x) = 1) : x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4045_404526


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l4045_404568

theorem rectangular_prism_volume (a b c : ℕ) 
  (h1 : 4 * ((a - 2) + (b - 2) + (c - 2)) = 40)
  (h2 : 2 * ((a - 2) * (b - 2) + (a - 2) * (c - 2) + (b - 2) * (c - 2)) = 66) :
  a * b * c = 150 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l4045_404568


namespace NUMINAMATH_CALUDE_pencil_profit_problem_l4045_404595

theorem pencil_profit_problem (total_pencils : ℕ) (buy_price sell_price : ℚ) (desired_profit : ℚ) :
  total_pencils = 1500 →
  buy_price = 1/10 →
  sell_price = 1/4 →
  desired_profit = 100 →
  ∃ (pencils_sold : ℕ), 
    pencils_sold ≤ total_pencils ∧
    sell_price * pencils_sold - buy_price * total_pencils = desired_profit ∧
    pencils_sold = 1000 :=
by sorry

end NUMINAMATH_CALUDE_pencil_profit_problem_l4045_404595


namespace NUMINAMATH_CALUDE_guards_per_team_is_five_l4045_404548

/-- The number of forwards in the league -/
def num_forwards : ℕ := 32

/-- The number of guards in the league -/
def num_guards : ℕ := 80

/-- The number of guards per team when creating the maximum number of teams -/
def guards_per_team : ℕ := num_guards / Nat.gcd num_forwards num_guards

theorem guards_per_team_is_five : guards_per_team = 5 := by
  sorry

end NUMINAMATH_CALUDE_guards_per_team_is_five_l4045_404548


namespace NUMINAMATH_CALUDE_adjacent_pair_arrangements_l4045_404591

def number_of_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial n

theorem adjacent_pair_arrangements :
  let total_people : ℕ := 5
  let adjacent_pair : ℕ := 2
  let remaining_people : ℕ := total_people - adjacent_pair + 1
  number_of_arrangements remaining_people remaining_people = 24 :=
by sorry

end NUMINAMATH_CALUDE_adjacent_pair_arrangements_l4045_404591


namespace NUMINAMATH_CALUDE_first_day_is_tuesday_l4045_404534

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Counts the occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem: In a 31-day month with exactly 4 Fridays and 4 Mondays, the first day is Tuesday -/
theorem first_day_is_tuesday (m : Month) 
  (h1 : m.days = 31)
  (h2 : countDayOccurrences m DayOfWeek.Friday = 4)
  (h3 : countDayOccurrences m DayOfWeek.Monday = 4) :
  m.firstDay = DayOfWeek.Tuesday :=
sorry

end NUMINAMATH_CALUDE_first_day_is_tuesday_l4045_404534


namespace NUMINAMATH_CALUDE_sock_selection_theorem_l4045_404507

theorem sock_selection_theorem (n : Nat) (k : Nat) (red : Nat) :
  n = 8 → k = 4 → red = 1 →
  (Nat.choose n k) - (Nat.choose (n - red) k) = 35 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_theorem_l4045_404507


namespace NUMINAMATH_CALUDE_sequence_property_implies_rational_factor_l4045_404572

/-- Two sequences are nonconstant if there exist two different terms -/
def Nonconstant (s : ℕ → ℚ) : Prop :=
  ∃ i j, i ≠ j ∧ s i ≠ s j

theorem sequence_property_implies_rational_factor
  (s t : ℕ → ℚ)
  (hs : Nonconstant s)
  (ht : Nonconstant t)
  (h : ∀ i j : ℕ, ∃ k : ℤ, (s i - s j) * (t i - t j) = k) :
  ∃ r : ℚ, (∀ i j : ℕ, ∃ m n : ℤ, (s i - s j) * r = m ∧ (t i - t j) / r = n) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_implies_rational_factor_l4045_404572


namespace NUMINAMATH_CALUDE_right_triangle_fraction_zero_smallest_constant_zero_l4045_404573

theorem right_triangle_fraction_zero (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  (a^2 + b^2 - c^2) / (a^2 + b^2 + c^2) = 0 := by
  sorry

theorem smallest_constant_zero :
  ∃ N, N = 0 ∧ ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a^2 + b^2 = c^2 →
    (a^2 + b^2 - c^2) / (a^2 + b^2 + c^2) < N := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_fraction_zero_smallest_constant_zero_l4045_404573


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l4045_404531

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the property of being monotonically increasing on [0,+∞)
def IsIncreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- Define the set of x satisfying f(2x-1) < f(3)
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | f (2*x - 1) < f 3}

-- State the theorem
theorem solution_set_is_open_interval
  (f : ℝ → ℝ) (h1 : IsEven f) (h2 : IsIncreasingOnNonnegative f) :
  SolutionSet f = Set.Ioo (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l4045_404531


namespace NUMINAMATH_CALUDE_solve_shooting_stars_l4045_404565

def shooting_stars_problem (bridget_count reginald_count sam_count : ℕ) : Prop :=
  bridget_count = 14 ∧
  reginald_count = bridget_count - 2 ∧
  sam_count = reginald_count + 4 ∧
  sam_count - ((bridget_count + reginald_count + sam_count) / 3) = 2

theorem solve_shooting_stars :
  ∃ (bridget_count reginald_count sam_count : ℕ),
    shooting_stars_problem bridget_count reginald_count sam_count :=
by
  sorry

end NUMINAMATH_CALUDE_solve_shooting_stars_l4045_404565


namespace NUMINAMATH_CALUDE_gcd_4830_3289_l4045_404525

theorem gcd_4830_3289 : Nat.gcd 4830 3289 = 23 := by
  sorry

end NUMINAMATH_CALUDE_gcd_4830_3289_l4045_404525


namespace NUMINAMATH_CALUDE_paint_cans_theorem_l4045_404567

/-- Represents the number of rooms that can be painted with one can of paint -/
def rooms_per_can : ℚ :=
  (36 - 28) / 4

/-- The number of cans used to paint 28 rooms -/
def cans_used : ℚ := 28 / rooms_per_can

theorem paint_cans_theorem :
  cans_used = 14 := by sorry

end NUMINAMATH_CALUDE_paint_cans_theorem_l4045_404567


namespace NUMINAMATH_CALUDE_lottery_is_event_l4045_404514

/-- Represents an experiment --/
inductive Experiment
  | TossCoin
  | Shoot
  | BoilWater
  | WinLottery

/-- Defines what constitutes an event --/
def is_event (e : Experiment) : Prop :=
  match e with
  | Experiment.WinLottery => True
  | _ => False

/-- Theorem stating that winning the lottery constitutes an event --/
theorem lottery_is_event : is_event Experiment.WinLottery := by
  sorry

end NUMINAMATH_CALUDE_lottery_is_event_l4045_404514


namespace NUMINAMATH_CALUDE_pizza_slices_remaining_l4045_404556

/-- The number of remaining pizza slices after eating some -/
def remaining_slices (slices_per_pizza : ℕ) (pizzas_ordered : ℕ) (slices_eaten : ℕ) : ℕ :=
  slices_per_pizza * pizzas_ordered - slices_eaten

/-- Theorem: Given the conditions, the number of remaining slices is 9 -/
theorem pizza_slices_remaining :
  remaining_slices 8 2 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_remaining_l4045_404556


namespace NUMINAMATH_CALUDE_subset_divisibility_property_l4045_404508

theorem subset_divisibility_property (A : Finset ℕ) (hA : A.card = 3) :
  ∃ (B : Finset ℕ) (x y : ℕ), B ⊆ A ∧ B.card = 2 ∧ x ∈ B ∧ y ∈ B ∧
    ∀ (m n : ℕ), Odd m → Odd n →
      (10 : ℤ) ∣ (((x ^ m : ℕ) * (y ^ n : ℕ)) - ((x ^ n : ℕ) * (y ^ m : ℕ))) :=
by sorry


end NUMINAMATH_CALUDE_subset_divisibility_property_l4045_404508


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_220_l4045_404532

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := 5 * circle_radius
  rectangle_length * rectangle_breadth

theorem rectangle_area_is_220 :
  rectangle_area 16 11 = 220 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_220_l4045_404532


namespace NUMINAMATH_CALUDE_initial_women_count_l4045_404544

def work_completion (women : ℕ) (children : ℕ) (days : ℕ) : Prop :=
  women * days + children * days = women * 7

theorem initial_women_count : ∃ x : ℕ,
  work_completion x 0 7 ∧
  work_completion 0 10 14 ∧
  work_completion 5 10 4 ∧
  x = 4 := by sorry

end NUMINAMATH_CALUDE_initial_women_count_l4045_404544


namespace NUMINAMATH_CALUDE_society_committee_selection_l4045_404535

theorem society_committee_selection (n : ℕ) (k : ℕ) : n = 20 ∧ k = 3 → Nat.choose n k = 1140 := by
  sorry

end NUMINAMATH_CALUDE_society_committee_selection_l4045_404535


namespace NUMINAMATH_CALUDE_no_real_solutions_l4045_404599

theorem no_real_solutions (k d : ℝ) (hk : k = -1) (hd : d < 0 ∨ d > 2) :
  ¬∃ (x y : ℝ), x^3 + y^3 = 2 ∧ y = k * x + d :=
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l4045_404599


namespace NUMINAMATH_CALUDE_domain_of_fourth_root_power_function_l4045_404559

-- Define the power function
def f (x : ℝ) : ℝ := x^(1/4)

-- State the theorem about the domain of f
theorem domain_of_fourth_root_power_function :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_domain_of_fourth_root_power_function_l4045_404559


namespace NUMINAMATH_CALUDE_alan_ticket_count_l4045_404552

theorem alan_ticket_count (alan marcy : ℕ) 
  (total : alan + marcy = 150)
  (marcy_relation : marcy = 5 * alan - 6) :
  alan = 26 := by
sorry

end NUMINAMATH_CALUDE_alan_ticket_count_l4045_404552


namespace NUMINAMATH_CALUDE_prob_two_hearts_is_one_seventeenth_l4045_404543

-- Define a standard deck of cards
def standard_deck : ℕ := 52

-- Define the number of hearts in a standard deck
def hearts_in_deck : ℕ := 13

-- Define the probability of drawing two hearts
def prob_two_hearts : ℚ := (hearts_in_deck : ℚ) / standard_deck * (hearts_in_deck - 1) / (standard_deck - 1)

-- Theorem statement
theorem prob_two_hearts_is_one_seventeenth : 
  prob_two_hearts = 1 / 17 := by sorry

end NUMINAMATH_CALUDE_prob_two_hearts_is_one_seventeenth_l4045_404543


namespace NUMINAMATH_CALUDE_birds_in_cage_l4045_404590

theorem birds_in_cage (birds_taken_out birds_left : ℕ) 
  (h1 : birds_taken_out = 10)
  (h2 : birds_left = 9) : 
  birds_taken_out + birds_left = 19 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_cage_l4045_404590


namespace NUMINAMATH_CALUDE_distribute_six_balls_two_boxes_l4045_404501

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to distribute 6 distinguishable balls into 2 distinguishable boxes is 64 -/
theorem distribute_six_balls_two_boxes : distribute_balls 6 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_two_boxes_l4045_404501


namespace NUMINAMATH_CALUDE_multiply_polynomial_equality_l4045_404530

theorem multiply_polynomial_equality (x : ℝ) : 
  (x^4 + 50 * x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomial_equality_l4045_404530


namespace NUMINAMATH_CALUDE_new_shape_perimeter_l4045_404584

/-- The perimeter of the new shape formed by cutting an isosceles triangle from a square and reattaching it outside --/
theorem new_shape_perimeter (square_perimeter : ℝ) (triangle_side : ℝ) : 
  square_perimeter = 64 →
  triangle_side = square_perimeter / 4 →
  square_perimeter / 4 + square_perimeter / 4 + square_perimeter / 4 + square_perimeter / 4 + square_perimeter / 4 = 80 :=
by sorry

end NUMINAMATH_CALUDE_new_shape_perimeter_l4045_404584


namespace NUMINAMATH_CALUDE_P_Q_disjoint_l4045_404522

def P : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}
def Q : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 2}

theorem P_Q_disjoint : P ∩ Q = ∅ := by
  sorry

end NUMINAMATH_CALUDE_P_Q_disjoint_l4045_404522


namespace NUMINAMATH_CALUDE_cally_white_shirts_l4045_404541

/-- Represents the number of clothes of each type for a person -/
structure ClothesCount where
  white_shirts : ℕ
  colored_shirts : ℕ
  shorts : ℕ
  pants : ℕ

/-- Calculates the total number of clothes for a person -/
def total_clothes (c : ClothesCount) : ℕ :=
  c.white_shirts + c.colored_shirts + c.shorts + c.pants

/-- Theorem: Cally washed 10 white shirts -/
theorem cally_white_shirts :
  ∀ (cally_clothes : ClothesCount),
    cally_clothes.colored_shirts = 5 →
    cally_clothes.shorts = 7 →
    cally_clothes.pants = 6 →
    ∀ (danny_clothes : ClothesCount),
      danny_clothes.white_shirts = 6 →
      danny_clothes.colored_shirts = 8 →
      danny_clothes.shorts = 10 →
      danny_clothes.pants = 6 →
      total_clothes cally_clothes + total_clothes danny_clothes = 58 →
      cally_clothes.white_shirts = 10 :=
by sorry

end NUMINAMATH_CALUDE_cally_white_shirts_l4045_404541


namespace NUMINAMATH_CALUDE_equation_equality_l4045_404592

theorem equation_equality : 3 * 6524 = 8254 * 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l4045_404592


namespace NUMINAMATH_CALUDE_rectangle_circles_inequality_l4045_404566

/-- Given a rectangle ABCD with sides a and b, and circles with radii r1 and r2
    as defined, prove that r1 + r2 ≥ 5/8 * (a + b). -/
theorem rectangle_circles_inequality (a b r1 r2 : ℝ) 
    (ha : a > 0) (hb : b > 0) (hr1 : r1 > 0) (hr2 : r2 > 0)
    (h_r1 : r1 = b / 2 + a^2 / (8 * b))
    (h_r2 : r2 = a / 2 + b^2 / (8 * a)) :
    r1 + r2 ≥ 5/8 * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circles_inequality_l4045_404566


namespace NUMINAMATH_CALUDE_painting_time_theorem_l4045_404563

def time_for_lily : ℕ := 5
def time_for_rose : ℕ := 7
def time_for_orchid : ℕ := 3
def time_for_vine : ℕ := 2

def num_lilies : ℕ := 17
def num_roses : ℕ := 10
def num_orchids : ℕ := 6
def num_vines : ℕ := 20

def total_time : ℕ := time_for_lily * num_lilies + time_for_rose * num_roses + 
                       time_for_orchid * num_orchids + time_for_vine * num_vines

theorem painting_time_theorem : total_time = 213 := by
  sorry

end NUMINAMATH_CALUDE_painting_time_theorem_l4045_404563


namespace NUMINAMATH_CALUDE_log_equation_solution_l4045_404579

theorem log_equation_solution (x : ℝ) :
  (Real.log x / Real.log 2) + (Real.log x / Real.log 8) = 5 →
  x = 2^(15/4) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l4045_404579


namespace NUMINAMATH_CALUDE_joan_initial_balloons_l4045_404596

/-- The number of blue balloons Joan initially had -/
def initial_balloons : ℕ := sorry

/-- The number of balloons Sally gave to Joan -/
def sally_gave : ℕ := 5

/-- The number of balloons Joan gave to Jessica -/
def joan_gave : ℕ := 2

/-- The number of balloons Joan has now -/
def joan_now : ℕ := 12

theorem joan_initial_balloons :
  initial_balloons + sally_gave - joan_gave = joan_now :=
sorry

end NUMINAMATH_CALUDE_joan_initial_balloons_l4045_404596


namespace NUMINAMATH_CALUDE_smallest_a_divisible_by_65_l4045_404505

theorem smallest_a_divisible_by_65 :
  ∃ (a : ℕ), a > 0 ∧ 
  (∀ (n : ℤ), 65 ∣ (5 * n^13 + 13 * n^5 + 9 * a * n)) ∧
  (∀ (b : ℕ), b > 0 → b < a → 
    ∃ (m : ℤ), ¬(65 ∣ (5 * m^13 + 13 * m^5 + 9 * b * m))) ∧
  a = 63 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_divisible_by_65_l4045_404505


namespace NUMINAMATH_CALUDE_stew_consumption_l4045_404528

theorem stew_consumption (total_stew : ℝ) : 
  let camp_fraction : ℝ := 1/3
  let range_fraction : ℝ := 1 - camp_fraction
  let lunch_consumption : ℝ := (1/4) * total_stew
  let evening_portion_multiplier : ℝ := 3/2
  let evening_consumption : ℝ := evening_portion_multiplier * (range_fraction / camp_fraction) * lunch_consumption
  lunch_consumption + evening_consumption = total_stew :=
by sorry

end NUMINAMATH_CALUDE_stew_consumption_l4045_404528


namespace NUMINAMATH_CALUDE_sequence_inequality_l4045_404503

-- Define a non-negative sequence
def non_negative_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, 0 ≤ a n

-- Define the condition for the sequence
def seq_condition (a : ℕ → ℝ) : Prop :=
  ∀ m n, a (m + n) ≤ a m + a n

-- State the theorem
theorem sequence_inequality (a : ℕ → ℝ) 
  (h_non_neg : non_negative_seq a) 
  (h_condition : seq_condition a) :
  ∀ m n, m ≤ n → a n ≤ m * a 1 + (n / m - 1) * a m :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l4045_404503


namespace NUMINAMATH_CALUDE_zoe_coloring_books_l4045_404537

/-- The number of pictures left to color given the initial number of pictures in two books and the number of pictures already colored. -/
def pictures_left_to_color (book1_pictures : ℕ) (book2_pictures : ℕ) (colored_pictures : ℕ) : ℕ :=
  book1_pictures + book2_pictures - colored_pictures

/-- Theorem stating that given two coloring books with 44 pictures each, and 20 pictures already colored, the number of pictures left to color is 68. -/
theorem zoe_coloring_books : pictures_left_to_color 44 44 20 = 68 := by
  sorry


end NUMINAMATH_CALUDE_zoe_coloring_books_l4045_404537


namespace NUMINAMATH_CALUDE_triangle_value_l4045_404509

theorem triangle_value (p : ℤ) (h1 : ∃ triangle : ℤ, triangle + p = 47) 
  (h2 : 3 * (47) - p = 133) : 
  ∃ triangle : ℤ, triangle = 39 :=
sorry

end NUMINAMATH_CALUDE_triangle_value_l4045_404509


namespace NUMINAMATH_CALUDE_min_balls_for_target_color_l4045_404506

def orange_balls : ℕ := 26
def purple_balls : ℕ := 21
def brown_balls : ℕ := 20
def gray_balls : ℕ := 15
def silver_balls : ℕ := 12
def golden_balls : ℕ := 10

def target_count : ℕ := 17

theorem min_balls_for_target_color :
  ∃ (n : ℕ), 
    (∀ (m : ℕ), m < n → 
      ∃ (o p b g s g' : ℕ), 
        o + p + b + g + s + g' = m ∧ 
        o ≤ orange_balls ∧ 
        p ≤ purple_balls ∧ 
        b ≤ brown_balls ∧ 
        g ≤ gray_balls ∧ 
        s ≤ silver_balls ∧ 
        g' ≤ golden_balls ∧
        o < target_count ∧ 
        p < target_count ∧ 
        b < target_count ∧ 
        g < target_count ∧ 
        s < target_count ∧ 
        g' < target_count) ∧
    (∀ (o p b g s g' : ℕ), 
      o + p + b + g + s + g' = n → 
      o ≤ orange_balls → 
      p ≤ purple_balls → 
      b ≤ brown_balls → 
      g ≤ gray_balls → 
      s ≤ silver_balls → 
      g' ≤ golden_balls →
      o ≥ target_count ∨ 
      p ≥ target_count ∨ 
      b ≥ target_count ∨ 
      g ≥ target_count ∨ 
      s ≥ target_count ∨ 
      g' ≥ target_count) ∧
    n = 86 :=
by sorry

end NUMINAMATH_CALUDE_min_balls_for_target_color_l4045_404506


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l4045_404545

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : average_children = 2)
  (h3 : childless_families = 3) :
  (total_families : ℚ) * average_children / ((total_families : ℚ) - childless_families) = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l4045_404545


namespace NUMINAMATH_CALUDE_parallel_vectors_problem_l4045_404511

/-- Given two vectors a and b in ℝ², where a = (1, -2), |b| = 2√5, and a is parallel to b,
    prove that b = (2, -4) or b = (-2, 4) -/
theorem parallel_vectors_problem (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (1, -2)
  (‖b‖ = 2 * Real.sqrt 5) →
  (∃ (k : ℝ), b = k • a) →
  (b = (2, -4) ∨ b = (-2, 4)) := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_problem_l4045_404511


namespace NUMINAMATH_CALUDE_johns_next_birthday_l4045_404524

-- Define variables for ages
variable (j b a : ℝ)

-- Define the relationships between ages
def john_bob_relation (j b : ℝ) : Prop := j = 1.25 * b
def bob_alice_relation (b a : ℝ) : Prop := b = 0.5 * a
def age_sum (j b a : ℝ) : Prop := j + b + a = 37.8

-- Theorem statement
theorem johns_next_birthday 
  (h1 : john_bob_relation j b)
  (h2 : bob_alice_relation b a)
  (h3 : age_sum j b a) :
  ⌈j⌉ + 1 = 12 := by sorry

end NUMINAMATH_CALUDE_johns_next_birthday_l4045_404524


namespace NUMINAMATH_CALUDE_sam_has_five_dimes_l4045_404523

/-- Represents the number of dimes Sam has at different stages -/
structure DimeCount where
  initial : ℕ
  after_sister_borrows : ℕ
  after_friend_borrows : ℕ
  after_sister_returns : ℕ
  after_friend_returns : ℕ

/-- Calculates the final number of dimes Sam has -/
def final_dime_count (d : DimeCount) : ℕ :=
  d.initial - 4 - 2 + 2 + 1

/-- Theorem stating that Sam ends up with 5 dimes -/
theorem sam_has_five_dimes (d : DimeCount) 
  (h_initial : d.initial = 8)
  (h_sister_borrows : d.after_sister_borrows = d.initial - 4)
  (h_friend_borrows : d.after_friend_borrows = d.after_sister_borrows - 2)
  (h_sister_returns : d.after_sister_returns = d.after_friend_borrows + 2)
  (h_friend_returns : d.after_friend_returns = d.after_sister_returns + 1) :
  final_dime_count d = 5 := by
  sorry

end NUMINAMATH_CALUDE_sam_has_five_dimes_l4045_404523


namespace NUMINAMATH_CALUDE_box_volume_cubes_l4045_404550

theorem box_volume_cubes (p : ℕ) (h : Prime p) : 
  p * (2 * p) * (3 * p) = 6 * p^3 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_cubes_l4045_404550


namespace NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l4045_404529

/-- Calculates the overall profit percentage for cricket bat sales -/
theorem cricket_bat_profit_percentage
  (num_a : ℕ) (price_a : ℚ) (profit_a : ℚ)
  (num_b : ℕ) (price_b : ℚ) (profit_b : ℚ) :
  num_a = 5 ∧ price_a = 850 ∧ profit_a = 225 ∧
  num_b = 10 ∧ price_b = 950 ∧ profit_b = 300 →
  let total_profit := num_a * profit_a + num_b * profit_b
  let total_revenue := num_a * price_a + num_b * price_b
  (total_profit / total_revenue) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l4045_404529


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l4045_404500

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 3 ∧ b = 4) ∨ (a = 3 ∧ c = 4) ∨ (b = 3 ∧ c = 4) →
  a^2 + b^2 = c^2 →
  c = 5 ∨ c = Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l4045_404500


namespace NUMINAMATH_CALUDE_trapezoid_area_example_l4045_404586

/-- Represents a trapezoid with sides a, b, c, d where a is parallel to c -/
structure Trapezoid :=
  (a b c d : ℝ)

/-- Calculates the area of a trapezoid -/
def trapezoidArea (t : Trapezoid) : ℝ :=
  sorry

theorem trapezoid_area_example :
  let t := Trapezoid.mk 52 20 65 11
  trapezoidArea t = 594 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_example_l4045_404586


namespace NUMINAMATH_CALUDE_equidistant_line_equation_l4045_404533

/-- A line passing through point (3,4) and equidistant from points (-2,2) and (4,-2) -/
structure EquidistantLine where
  -- The line passes through point (3,4)
  passes_through : ℝ → ℝ → Prop
  -- The line is equidistant from points (-2,2) and (4,-2)
  equidistant : ℝ → ℝ → Prop

/-- The equation of the line is either 2x-y-2=0 or 2x+3y-18=0 -/
def line_equation (l : EquidistantLine) : Prop :=
  (∀ x y, l.passes_through x y ∧ l.equidistant x y → 2*x - y - 2 = 0) ∨
  (∀ x y, l.passes_through x y ∧ l.equidistant x y → 2*x + 3*y - 18 = 0)

theorem equidistant_line_equation (l : EquidistantLine) : line_equation l := by
  sorry

end NUMINAMATH_CALUDE_equidistant_line_equation_l4045_404533


namespace NUMINAMATH_CALUDE_total_grains_in_grey_parts_l4045_404577

/-- Represents a circle with grains -/
structure GrainCircle where
  total : ℕ
  nonOverlapping : ℕ

/-- Calculates the number of grains in the overlapping part of a circle -/
def overlappingGrains (circle : GrainCircle) : ℕ :=
  circle.total - circle.nonOverlapping

/-- Represents two overlapping circles with grains -/
structure OverlappingCircles where
  circle1 : GrainCircle
  circle2 : GrainCircle

/-- Theorem: The total number of grains in both grey parts is 61 -/
theorem total_grains_in_grey_parts (circles : OverlappingCircles)
  (h1 : circles.circle1.total = 87)
  (h2 : circles.circle2.total = 110)
  (h3 : circles.circle1.nonOverlapping = 68)
  (h4 : circles.circle2.nonOverlapping = 68) :
  overlappingGrains circles.circle1 + overlappingGrains circles.circle2 = 61 := by
  sorry

end NUMINAMATH_CALUDE_total_grains_in_grey_parts_l4045_404577


namespace NUMINAMATH_CALUDE_additive_inverses_imply_x_equals_one_l4045_404557

theorem additive_inverses_imply_x_equals_one :
  ∀ x : ℝ, (4 * x - 1) + (3 * x - 6) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_additive_inverses_imply_x_equals_one_l4045_404557


namespace NUMINAMATH_CALUDE_max_value_of_expression_l4045_404582

theorem max_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → -1/(2*x) - 2/y ≤ -1/(2*a) - 2/b) ∧
  (-1/(2*a) - 2/b = -9/2) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l4045_404582


namespace NUMINAMATH_CALUDE_solution_set_inequality_l4045_404571

theorem solution_set_inequality (m : ℝ) (h : m < 5) :
  {x : ℝ | m * x > 6 * x + 3} = {x : ℝ | x < 3 / (m - 6)} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l4045_404571


namespace NUMINAMATH_CALUDE_vector_norm_inequality_l4045_404558

theorem vector_norm_inequality (a₁ a₂ b₁ b₂ : ℝ) :
  Real.sqrt (a₁^2 + a₂^2) + Real.sqrt (b₁^2 + b₂^2) ≥ Real.sqrt ((a₁ - b₁)^2 + (a₂ - b₂)^2) := by
  sorry

end NUMINAMATH_CALUDE_vector_norm_inequality_l4045_404558


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l4045_404540

/-- A quadratic radical is considered "simple" if it cannot be simplified further. -/
def IsSimpleQuadraticRadical (x : ℝ) : Prop :=
  x ≥ 0 ∧ ∀ y z : ℝ, x = y * y * z → y = 1 ∨ z < 0

/-- The set of quadratic radicals to consider -/
def QuadraticRadicals : Set ℝ := {4, 7, 12, 0.5}

theorem simplest_quadratic_radical :
  ∃ (x : ℝ), x ∈ QuadraticRadicals ∧
    IsSimpleQuadraticRadical (Real.sqrt x) ∧
    ∀ y ∈ QuadraticRadicals, IsSimpleQuadraticRadical (Real.sqrt y) → y = x :=
by
  sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l4045_404540


namespace NUMINAMATH_CALUDE_rolling_circle_arc_angle_range_l4045_404575

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the arc angle given three points on a circle -/
def arcAngle (circle : Circle) (p1 p2 p3 : Point) : ℝ := sorry

/-- Main theorem -/
theorem rolling_circle_arc_angle_range 
  (triangle : Triangle) 
  (isRightTriangle : angle triangle.A triangle.B triangle.C = 90)
  (has30DegreeAngle : angle triangle.A triangle.C triangle.B = 30)
  (circle : Circle)
  (circleRadiusHalfBC : circle.radius = distance triangle.B triangle.C / 2)
  (T : Point → Point) -- T is a function of the circle's position
  (M : Point → Point) -- M is a function of the circle's position
  (N : Point → Point) -- N is a function of the circle's position
  (circleTangentToAB : ∀ (circlePos : Point), distance (T circlePos) circlePos = circle.radius)
  (circleIntersectsAC : ∀ (circlePos : Point), (M circlePos).x = triangle.A.x ∨ (M circlePos).y = triangle.A.y)
  (circleIntersectsBC : ∀ (circlePos : Point), distance (N circlePos) triangle.B = distance (N circlePos) triangle.C) :
  ∃ (circlePos1 circlePos2 : Point),
    arcAngle circle (M circlePos1) (T circlePos1) (N circlePos1) = 180 ∧
    arcAngle circle (M circlePos2) (T circlePos2) (N circlePos2) = 0 ∧
    ∀ (circlePos : Point),
      0 ≤ arcAngle circle (M circlePos) (T circlePos) (N circlePos) ∧
      arcAngle circle (M circlePos) (T circlePos) (N circlePos) ≤ 180 :=
sorry

end NUMINAMATH_CALUDE_rolling_circle_arc_angle_range_l4045_404575


namespace NUMINAMATH_CALUDE_perpendicular_line_through_intersection_l4045_404583

/-- Given a line l with equation 2x - y - 4 = 0, prove that the line with equation
    x + 2y - 2 = 0 is perpendicular to l and passes through the point where l
    intersects the x-axis. -/
theorem perpendicular_line_through_intersection (x y : ℝ) : 
  let l : ℝ → ℝ → Prop := λ x y ↦ 2 * x - y - 4 = 0
  let m : ℝ × ℝ := (2, 0)  -- Intersection point of l with x-axis
  let perp : ℝ → ℝ → Prop := λ x y ↦ x + 2 * y - 2 = 0
  (∀ x y, l x y → (x - m.1) * (x - m.1) + (y - m.2) * (y - m.2) ≠ 0 →
    (perp x y ↔ (x - m.1) * (2) + (y - m.2) * (-1) = 0)) ∧
  perp m.1 m.2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_intersection_l4045_404583


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l4045_404516

theorem sum_of_two_numbers (a b : ℕ) : a = 30 ∧ b = 42 ∧ b = a + 12 → a + b = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l4045_404516


namespace NUMINAMATH_CALUDE_min_value_x_minus_inv_y_l4045_404527

theorem min_value_x_minus_inv_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 3 * x + y + 1 / x + 2 / y = 13 / 2) :
  x - 1 / y ≥ -1 / 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
  3 * x₀ + y₀ + 1 / x₀ + 2 / y₀ = 13 / 2 ∧ x₀ - 1 / y₀ = -1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_minus_inv_y_l4045_404527


namespace NUMINAMATH_CALUDE_percentage_of_material_A_in_first_solution_l4045_404587

/-- Given two solutions and their mixture, proves the percentage of material A in the first solution -/
theorem percentage_of_material_A_in_first_solution 
  (x : ℝ) -- Percentage of material A in the first solution
  (h1 : x + 80 = 100) -- First solution composition
  (h2 : 30 + 70 = 100) -- Second solution composition
  (h3 : 0.8 * x + 0.2 * 30 = 22) -- Mixture composition
  : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_material_A_in_first_solution_l4045_404587


namespace NUMINAMATH_CALUDE_determinant_zero_l4045_404519

theorem determinant_zero (x y z : ℝ) : 
  Matrix.det !![1, x, y+z; 1, x+y, z; 1, x+z, y] = 0 := by
  sorry

end NUMINAMATH_CALUDE_determinant_zero_l4045_404519


namespace NUMINAMATH_CALUDE_optimal_distribution_is_best_l4045_404581

/-- Represents the distribution of coins proposed by a logician -/
structure Distribution :=
  (logician1 : ℕ)
  (logician2 : ℕ)
  (logician3 : ℕ)

/-- The total number of coins to be distributed -/
def totalCoins : ℕ := 10

/-- The number of coins given to an eliminated logician -/
def eliminationCoins : ℕ := 2

/-- Checks if a distribution is valid (sums to the total number of coins) -/
def isValidDistribution (d : Distribution) : Prop :=
  d.logician1 + d.logician2 + d.logician3 = totalCoins

/-- Represents the approval of a distribution by a logician -/
def approves (logician : ℕ) (d : Distribution) : Prop :=
  match logician with
  | 1 => d.logician1 ≥ eliminationCoins
  | 2 => d.logician2 ≥ eliminationCoins
  | 3 => d.logician3 ≥ eliminationCoins
  | _ => False

/-- Checks if a distribution receives majority approval -/
def hasApproval (d : Distribution) : Prop :=
  (approves 1 d ∧ (approves 2 d ∨ approves 3 d)) ∨
  (approves 2 d ∧ approves 3 d)

/-- The optimal distribution strategy for Logician 1 -/
def optimalDistribution : Distribution :=
  { logician1 := 9, logician2 := 0, logician3 := 1 }

/-- Theorem stating that the optimal distribution is valid and maximizes Logician 1's gain -/
theorem optimal_distribution_is_best :
  isValidDistribution optimalDistribution ∧
  hasApproval optimalDistribution ∧
  ∀ d : Distribution,
    isValidDistribution d ∧ hasApproval d →
    d.logician1 ≤ optimalDistribution.logician1 :=
sorry


end NUMINAMATH_CALUDE_optimal_distribution_is_best_l4045_404581


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l4045_404549

/-- The equation of a conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 - 4*y^2 - 10*x + 20*y + 25 = 0

/-- Definition of a hyperbola -/
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧
    ∀ x y, f x y ↔ ((x - h) / a)^2 - ((y - k) / b)^2 = 1

/-- Theorem: The given equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l4045_404549


namespace NUMINAMATH_CALUDE_multiplication_problem_l4045_404517

theorem multiplication_problem : ∃ x : ℕ, 987 * x = 555681 ∧ x = 563 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l4045_404517


namespace NUMINAMATH_CALUDE_points_per_round_l4045_404504

def total_points : ℕ := 300
def num_rounds : ℕ := 5

theorem points_per_round :
  ∃ (points_per_round : ℕ),
    points_per_round * num_rounds = total_points ∧
    points_per_round = 60 :=
by sorry

end NUMINAMATH_CALUDE_points_per_round_l4045_404504


namespace NUMINAMATH_CALUDE_library_books_count_l4045_404510

/-- The number of books in a library after two years of purchases -/
def total_books (initial : ℕ) (last_year : ℕ) (multiplier : ℕ) : ℕ :=
  initial + last_year + multiplier * last_year

/-- Theorem stating the total number of books in the library -/
theorem library_books_count : total_books 100 50 3 = 300 := by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l4045_404510


namespace NUMINAMATH_CALUDE_prob_five_candy_is_thirty_percent_l4045_404547

-- Define the total number of eggs (can be any positive integer)
variable (total_eggs : ℕ) (h_total : total_eggs > 0)

-- Define the fractions of blue and purple eggs
def blue_fraction : ℚ := 4/5
def purple_fraction : ℚ := 1/5

-- Define the fractions of blue and purple eggs with 5 pieces of candy
def blue_five_candy_fraction : ℚ := 1/4
def purple_five_candy_fraction : ℚ := 1/2

-- Define the probability of getting 5 pieces of candy
def prob_five_candy : ℚ := blue_fraction * blue_five_candy_fraction + purple_fraction * purple_five_candy_fraction

-- Theorem: The probability of getting 5 pieces of candy is 30%
theorem prob_five_candy_is_thirty_percent : prob_five_candy = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_prob_five_candy_is_thirty_percent_l4045_404547


namespace NUMINAMATH_CALUDE_quadratic_roots_arithmetic_progression_l4045_404562

theorem quadratic_roots_arithmetic_progression 
  (a b c : ℝ) 
  (p₁ p₂ q₁ q₂ : ℝ) 
  (h₁ : a * p₁^2 + b * p₁ + c = 0)
  (h₂ : a * p₂^2 + b * p₂ + c = 0)
  (h₃ : c * q₁^2 + b * q₁ + a = 0)
  (h₄ : c * q₂^2 + b * q₂ + a = 0)
  (h₅ : p₁ ≠ p₂)
  (h₆ : q₁ ≠ q₂)
  (h₇ : ∃ d : ℝ, q₁ = p₁ + d ∧ p₂ = p₁ + 2*d ∧ q₂ = p₁ + 3*d) :
  a + c = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_arithmetic_progression_l4045_404562


namespace NUMINAMATH_CALUDE_max_amount_received_back_l4045_404570

/-- Represents the result of a gambling session -/
structure GamblingResult where
  initial_amount : ℕ
  chip_30_value : ℕ
  chip_100_value : ℕ
  total_chips_lost : ℕ
  chip_30_lost : ℕ
  chip_100_lost : ℕ

/-- Calculates the amount received back after a gambling session -/
def amount_received_back (result : GamblingResult) : ℕ :=
  result.initial_amount - (result.chip_30_lost * result.chip_30_value + result.chip_100_lost * result.chip_100_value)

/-- Theorem stating the maximum amount received back under given conditions -/
theorem max_amount_received_back :
  ∀ (result : GamblingResult),
    result.initial_amount = 3000 ∧
    result.chip_30_value = 30 ∧
    result.chip_100_value = 100 ∧
    result.total_chips_lost = 16 ∧
    (result.chip_30_lost = result.chip_100_lost + 2 ∨ result.chip_30_lost = result.chip_100_lost - 2) →
    amount_received_back result ≤ 1890 :=
by sorry

end NUMINAMATH_CALUDE_max_amount_received_back_l4045_404570
