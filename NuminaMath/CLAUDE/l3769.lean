import Mathlib

namespace NUMINAMATH_CALUDE_compare_magnitudes_l3769_376901

theorem compare_magnitudes (a b : ℝ) (ha : a ≠ 1) :
  a^2 + b^2 > 2*(a - b - 1) := by
sorry

end NUMINAMATH_CALUDE_compare_magnitudes_l3769_376901


namespace NUMINAMATH_CALUDE_midpoint_of_AB_l3769_376931

-- Define the point F
def F : ℝ × ℝ := (0, 1)

-- Define the line y = -5
def line_y_neg5 (x : ℝ) : ℝ := -5

-- Define the line x - 4y + 2 = 0
def line_l (x y : ℝ) : Prop := x - 4*y + 2 = 0

-- Define the distance condition for point P
def distance_condition (P : ℝ × ℝ) : Prop :=
  ∃ (d : ℝ), d > 0 ∧ 
    (Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) + 4 = Real.sqrt ((P.1 - P.1)^2 + (P.2 - line_y_neg5 P.1)^2))

-- Define the trajectory of P (parabola)
def trajectory (x y : ℝ) : Prop := x^2 = 4*y

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  trajectory A.1 A.2 ∧ trajectory B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem midpoint_of_AB :
  ∀ (P A B : ℝ × ℝ),
  distance_condition P →
  intersection_points A B →
  (A.1 + B.1) / 2 = 1/2 ∧ (A.2 + B.2) / 2 = 5/8 :=
sorry

end NUMINAMATH_CALUDE_midpoint_of_AB_l3769_376931


namespace NUMINAMATH_CALUDE_expression_values_l3769_376990

theorem expression_values (a b : ℝ) : 
  (∀ x : ℝ, |a| ≤ |x|) → (b * b = 1) → 
  (|a - 2| - b^2023 = 1 ∨ |a - 2| - b^2023 = 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l3769_376990


namespace NUMINAMATH_CALUDE_carole_wins_iff_n_odd_l3769_376988

/-- The game interval -/
def GameInterval (n : ℕ) := Set.Icc (0 : ℝ) n

/-- Predicate for a valid move -/
def ValidMove (prev : Set ℝ) (x : ℝ) : Prop :=
  ∀ y ∈ prev, |x - y| ≥ 1.5

/-- The game state -/
structure GameState (n : ℕ) where
  chosen : Set ℝ
  current_player : Bool -- true for Carole, false for Leo

/-- The game result -/
inductive GameResult
  | CaroleWins
  | LeoWins

/-- Optimal strategy -/
def OptimalStrategy (n : ℕ) : GameState n → GameResult :=
  sorry

/-- The main theorem -/
theorem carole_wins_iff_n_odd (n : ℕ) (h : n > 10) :
  OptimalStrategy n { chosen := ∅, current_player := true } = GameResult.CaroleWins ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_carole_wins_iff_n_odd_l3769_376988


namespace NUMINAMATH_CALUDE_stating_max_remaining_coins_product_mod_l3769_376970

/-- Represents the grid size -/
def gridSize : Nat := 418

/-- Represents the modulus for the final result -/
def modulus : Nat := 2007

/-- Represents the maximum number of coins that can remain in one quadrant -/
def maxCoinsPerQuadrant : Nat := (gridSize / 2) * (gridSize / 2)

/-- 
Theorem stating that the maximum value of bw (mod 2007) is 1999, 
where b and w are the number of remaining black and white coins respectively 
after applying the removal rules on a 418 × 418 grid.
-/
theorem max_remaining_coins_product_mod (b w : Nat) : 
  b ≤ maxCoinsPerQuadrant → 
  w ≤ maxCoinsPerQuadrant → 
  (b * w) % modulus ≤ 1999 ∧ 
  ∃ (b' w' : Nat), b' ≤ maxCoinsPerQuadrant ∧ w' ≤ maxCoinsPerQuadrant ∧ (b' * w') % modulus = 1999 := by
  sorry

end NUMINAMATH_CALUDE_stating_max_remaining_coins_product_mod_l3769_376970


namespace NUMINAMATH_CALUDE_line_point_order_l3769_376914

/-- Given a line y = mx + n where m < 0 and n > 0, if points A(-2, y₁), B(-3, y₂), and C(1, y₃) 
    are on the line, then y₃ < y₁ < y₂. -/
theorem line_point_order (m n y₁ y₂ y₃ : ℝ) 
    (hm : m < 0) (hn : n > 0)
    (hA : y₁ = m * (-2) + n)
    (hB : y₂ = m * (-3) + n)
    (hC : y₃ = m * 1 + n) :
  y₃ < y₁ ∧ y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_line_point_order_l3769_376914


namespace NUMINAMATH_CALUDE_smallest_symmetric_set_l3769_376999

-- Define a point in the xy-plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the set T
def T : Set Point := sorry

-- Define symmetry conditions
def symmetricAboutOrigin (p : Point) : Prop :=
  Point.mk (-p.x) (-p.y) ∈ T

def symmetricAboutXAxis (p : Point) : Prop :=
  Point.mk p.x (-p.y) ∈ T

def symmetricAboutYAxis (p : Point) : Prop :=
  Point.mk (-p.x) p.y ∈ T

def symmetricAboutNegativeDiagonal (p : Point) : Prop :=
  Point.mk (-p.y) (-p.x) ∈ T

-- State the theorem
theorem smallest_symmetric_set :
  (∀ p ∈ T, symmetricAboutOrigin p ∧ 
            symmetricAboutXAxis p ∧ 
            symmetricAboutYAxis p ∧ 
            symmetricAboutNegativeDiagonal p) →
  Point.mk 1 4 ∈ T →
  (∃ (s : Finset Point), s.card = 8 ∧ ↑s = T) ∧
  ¬∃ (s : Finset Point), s.card < 8 ∧ ↑s = T :=
by sorry

end NUMINAMATH_CALUDE_smallest_symmetric_set_l3769_376999


namespace NUMINAMATH_CALUDE_power_inequality_l3769_376943

theorem power_inequality (x y a b : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : a > b) (h4 : b > 1) :
  a^x > b^y := by sorry

end NUMINAMATH_CALUDE_power_inequality_l3769_376943


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l3769_376915

theorem opposite_of_negative_two_thirds :
  let x : ℚ := -2/3
  let opposite (y : ℚ) := -y
  opposite x = 2/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l3769_376915


namespace NUMINAMATH_CALUDE_ducks_arrived_later_l3769_376948

theorem ducks_arrived_later (initial_ducks : ℕ) (initial_geese : ℕ) (final_ducks : ℕ) (final_geese : ℕ) : 
  initial_ducks = 25 →
  initial_geese = 2 * initial_ducks - 10 →
  final_geese = initial_geese - (15 - 5) →
  final_geese = final_ducks + 1 →
  final_ducks - initial_ducks = 4 :=
by sorry

end NUMINAMATH_CALUDE_ducks_arrived_later_l3769_376948


namespace NUMINAMATH_CALUDE_toys_produced_daily_l3769_376980

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 3400

/-- The number of working days per week -/
def working_days : ℕ := 5

/-- The number of toys produced each day -/
def toys_per_day : ℕ := toys_per_week / working_days

/-- Theorem stating that the number of toys produced each day is 680 -/
theorem toys_produced_daily : toys_per_day = 680 := by
  sorry

end NUMINAMATH_CALUDE_toys_produced_daily_l3769_376980


namespace NUMINAMATH_CALUDE_range_of_a_l3769_376987

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x - 10 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}
def C (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a + 3}

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (((A ∩ B) ∩ C a) = C a) ↔ 1 ≤ a := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3769_376987


namespace NUMINAMATH_CALUDE_leak_drain_time_l3769_376950

-- Define the pump filling rate
def pump_rate : ℚ := 1 / 2

-- Define the time to fill with leak
def fill_time_with_leak : ℚ := 7 / 3

-- Define the combined rate (pump - leak)
def combined_rate : ℚ := 1 / fill_time_with_leak

-- Define the leak rate
def leak_rate : ℚ := pump_rate - combined_rate

-- Theorem statement
theorem leak_drain_time (pump_rate : ℚ) (fill_time_with_leak : ℚ) 
  (combined_rate : ℚ) (leak_rate : ℚ) :
  pump_rate = 1 / 2 →
  fill_time_with_leak = 7 / 3 →
  combined_rate = 1 / fill_time_with_leak →
  leak_rate = pump_rate - combined_rate →
  1 / leak_rate = 14 := by
  sorry

end NUMINAMATH_CALUDE_leak_drain_time_l3769_376950


namespace NUMINAMATH_CALUDE_total_savings_percentage_l3769_376913

/-- Calculates the total savings percentage given the original prices and discount rates -/
theorem total_savings_percentage
  (jacket_price shirt_price hat_price : ℝ)
  (jacket_discount shirt_discount hat_discount : ℝ)
  (h_jacket_price : jacket_price = 100)
  (h_shirt_price : shirt_price = 50)
  (h_hat_price : hat_price = 30)
  (h_jacket_discount : jacket_discount = 0.3)
  (h_shirt_discount : shirt_discount = 0.6)
  (h_hat_discount : hat_discount = 0.5) :
  (jacket_price * jacket_discount + shirt_price * shirt_discount + hat_price * hat_discount) /
  (jacket_price + shirt_price + hat_price) * 100 = 41.67 :=
by sorry

end NUMINAMATH_CALUDE_total_savings_percentage_l3769_376913


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_even_integers_l3769_376992

theorem sum_of_four_consecutive_even_integers :
  ¬ (∃ m : ℤ, 56 = 4*m + 12 ∧ Even m) ∧
  (∃ m : ℤ, 20 = 4*m + 12 ∧ Even m) ∧
  (∃ m : ℤ, 108 = 4*m + 12 ∧ Even m) ∧
  (∃ m : ℤ, 88 = 4*m + 12 ∧ Even m) ∧
  (∃ m : ℤ, 200 = 4*m + 12 ∧ Even m) := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_even_integers_l3769_376992


namespace NUMINAMATH_CALUDE_shoes_polished_percentage_l3769_376986

def shoes_polished (pairs : ℕ) (left_to_polish : ℕ) : ℚ :=
  let total_shoes := 2 * pairs
  let polished := total_shoes - left_to_polish
  (polished : ℚ) / (total_shoes : ℚ) * 100

theorem shoes_polished_percentage :
  shoes_polished 10 11 = 45 := by
  sorry

end NUMINAMATH_CALUDE_shoes_polished_percentage_l3769_376986


namespace NUMINAMATH_CALUDE_sum_30_to_40_proof_l3769_376906

def sum_30_to_40 : ℕ := (List.range 11).map (· + 30) |>.sum

def even_count_30_to_40 : ℕ := (List.range 11).map (· + 30) |>.filter (· % 2 = 0) |>.length

theorem sum_30_to_40_proof : sum_30_to_40 = 385 :=
  by
  have h1 : sum_30_to_40 + even_count_30_to_40 = 391 := by sorry
  sorry

#eval sum_30_to_40
#eval even_count_30_to_40

end NUMINAMATH_CALUDE_sum_30_to_40_proof_l3769_376906


namespace NUMINAMATH_CALUDE_expression_evaluation_1_expression_evaluation_2_l3769_376958

theorem expression_evaluation_1 : (1 * (-4.5) - (-5 - (2/3)) - 2.5 - (7 + (2/3))) = -9 := by sorry

theorem expression_evaluation_2 : (-4^2 / (-2)^3 - (4/9) * (-3/2)^2) = 1 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_1_expression_evaluation_2_l3769_376958


namespace NUMINAMATH_CALUDE_scientific_notation_56_9_billion_l3769_376976

def billion : ℝ := 1000000000

theorem scientific_notation_56_9_billion :
  56.9 * billion = 5.69 * (10 : ℝ) ^ 9 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_56_9_billion_l3769_376976


namespace NUMINAMATH_CALUDE_fraction_equality_l3769_376954

theorem fraction_equality (w z : ℝ) (h : (1/w + 1/z)/(1/w - 1/z) = 1001) : (w + z)/(w - z) = -501 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3769_376954


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l3769_376994

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 40 →
  football = 26 →
  tennis = 20 →
  neither = 11 →
  ∃ both : ℕ, both = 17 ∧ total = football + tennis - both + neither :=
by sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l3769_376994


namespace NUMINAMATH_CALUDE_quadratic_function_m_value_l3769_376973

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (A B C : ℤ) : ℝ → ℝ := fun x ↦ A * x^2 + B * x + C

theorem quadratic_function_m_value
  (A B C : ℤ)
  (h1 : QuadraticFunction A B C 2 = 0)
  (h2 : 100 < QuadraticFunction A B C 9 ∧ QuadraticFunction A B C 9 < 110)
  (h3 : 150 < QuadraticFunction A B C 10 ∧ QuadraticFunction A B C 10 < 160)
  (h4 : ∃ m : ℤ, 10000 * m < QuadraticFunction A B C 200 ∧ QuadraticFunction A B C 200 < 10000 * (m + 1)) :
  ∃ m : ℤ, m = 16 ∧ 10000 * m < QuadraticFunction A B C 200 ∧ QuadraticFunction A B C 200 < 10000 * (m + 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_m_value_l3769_376973


namespace NUMINAMATH_CALUDE_addition_to_reach_target_l3769_376951

theorem addition_to_reach_target : (1250 / 50) + 7500 = 7525 := by
  sorry

end NUMINAMATH_CALUDE_addition_to_reach_target_l3769_376951


namespace NUMINAMATH_CALUDE_base4_division_theorem_l3769_376902

/-- Converts a number from base 4 to base 10 --/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- Performs division in base 4 --/
def divBase4 (a b : ℕ) : ℕ := sorry

theorem base4_division_theorem :
  let dividend := 1302
  let divisor := 12
  let quotient := 103
  divBase4 dividend divisor = quotient := by sorry

end NUMINAMATH_CALUDE_base4_division_theorem_l3769_376902


namespace NUMINAMATH_CALUDE_hannah_easter_eggs_l3769_376949

theorem hannah_easter_eggs :
  ∀ (total helen hannah : ℕ),
  total = 63 →
  hannah = 2 * helen →
  total = helen + hannah →
  hannah = 42 := by
sorry

end NUMINAMATH_CALUDE_hannah_easter_eggs_l3769_376949


namespace NUMINAMATH_CALUDE_sum_lent_is_400_l3769_376974

/-- Prove that the sum lent is 400, given the conditions of the problem -/
theorem sum_lent_is_400 
  (interest_rate : ℚ) 
  (time_period : ℕ) 
  (interest_difference : ℚ) 
  (h1 : interest_rate = 4 / 100)
  (h2 : time_period = 8)
  (h3 : interest_difference = 272) :
  ∃ (sum_lent : ℚ), 
    sum_lent * interest_rate * time_period = sum_lent - interest_difference ∧ 
    sum_lent = 400 := by
  sorry

end NUMINAMATH_CALUDE_sum_lent_is_400_l3769_376974


namespace NUMINAMATH_CALUDE_table_tennis_match_results_l3769_376917

/-- Represents a "best-of-3" table tennis match -/
structure TableTennisMatch where
  prob_a_win : ℝ
  prob_b_win : ℝ

/-- The probability of player A winning a single game -/
def prob_a_win (m : TableTennisMatch) : ℝ := m.prob_a_win

/-- The probability of player B winning a single game -/
def prob_b_win (m : TableTennisMatch) : ℝ := m.prob_b_win

/-- The probability of player A winning the entire match -/
def prob_a_win_match (m : TableTennisMatch) : ℝ :=
  (m.prob_a_win)^2 + 2 * m.prob_b_win * (m.prob_a_win)^2

/-- The expected number of games won by player A -/
def expected_games_won_a (m : TableTennisMatch) : ℝ :=
  1 * (2 * m.prob_a_win * (m.prob_b_win)^2) + 2 * ((m.prob_a_win)^2 + 2 * m.prob_b_win * (m.prob_a_win)^2)

/-- The variance of the number of games won by player A -/
def variance_games_won_a (m : TableTennisMatch) : ℝ :=
  (m.prob_b_win)^2 * (0 - expected_games_won_a m)^2 +
  (2 * m.prob_a_win * (m.prob_b_win)^2) * (1 - expected_games_won_a m)^2 +
  ((m.prob_a_win)^2 + 2 * m.prob_b_win * (m.prob_a_win)^2) * (2 - expected_games_won_a m)^2

theorem table_tennis_match_results (m : TableTennisMatch) 
  (h1 : m.prob_a_win = 0.6) 
  (h2 : m.prob_b_win = 0.4) : 
  prob_a_win_match m = 0.648 ∧ 
  expected_games_won_a m = 1.5 ∧ 
  variance_games_won_a m = 0.57 := by
  sorry

end NUMINAMATH_CALUDE_table_tennis_match_results_l3769_376917


namespace NUMINAMATH_CALUDE_locus_of_p_forms_two_circles_l3769_376904

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the circle
structure PointOnCircle (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

-- Define the projection of a point onto a line
def project_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ × ℝ :=
  sorry

-- Define the point P on OQ such that OP = QQ'
def point_p (c : Circle) (q : PointOnCircle c) (diameter : ℝ × ℝ → ℝ) : ℝ × ℝ :=
  sorry

-- Theorem stating that the locus of P forms two circles
theorem locus_of_p_forms_two_circles (c : Circle) (diameter : ℝ × ℝ → ℝ) :
  ∃ (c1 c2 : Circle),
    ∀ (q : PointOnCircle c),
      let p := point_p c q diameter
      (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∨
      (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2 :=
sorry

end NUMINAMATH_CALUDE_locus_of_p_forms_two_circles_l3769_376904


namespace NUMINAMATH_CALUDE_box_value_proof_l3769_376940

theorem box_value_proof : ∃ (square : ℚ), (6 + square) / 20 = 1 / 2 :=
  by
    use 4
    sorry

end NUMINAMATH_CALUDE_box_value_proof_l3769_376940


namespace NUMINAMATH_CALUDE_baker_loaves_per_hour_l3769_376923

/-- The number of ovens the baker has -/
def num_ovens : ℕ := 4

/-- The number of hours the baker bakes on weekdays -/
def weekday_hours : ℕ := 5

/-- The number of hours the baker bakes on weekend days -/
def weekend_hours : ℕ := 2

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The number of weeks the baker bakes -/
def num_weeks : ℕ := 3

/-- The total number of loaves baked in 3 weeks -/
def total_loaves : ℕ := 1740

/-- The number of loaves baked per hour in one oven -/
def loaves_per_hour : ℚ := 5

theorem baker_loaves_per_hour :
  loaves_per_hour = total_loaves / (num_ovens * (weekdays * weekday_hours + weekend_days * weekend_hours) * num_weeks) :=
by sorry

end NUMINAMATH_CALUDE_baker_loaves_per_hour_l3769_376923


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3769_376934

/-- The equation represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  (k - 3) * (k + 3) > 0

/-- The condition k > 3 is sufficient for the equation to represent a hyperbola -/
theorem sufficient_condition (k : ℝ) (h : k > 3) : is_hyperbola k := by sorry

/-- The condition k > 3 is not necessary for the equation to represent a hyperbola -/
theorem not_necessary_condition : ∃ k, k ≤ 3 ∧ is_hyperbola k := by sorry

/-- k > 3 is a sufficient but not necessary condition for the equation to represent a hyperbola -/
theorem sufficient_but_not_necessary : 
  (∀ k, k > 3 → is_hyperbola k) ∧ (∃ k, k ≤ 3 ∧ is_hyperbola k) := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3769_376934


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3769_376972

theorem no_integer_solutions : ¬∃ (x y : ℤ), (x ≠ 0 ∧ y ≠ 0) ∧ (x^2 / y - y^2 / x = 3 * (2 + 1 / (x * y))) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3769_376972


namespace NUMINAMATH_CALUDE_plumber_copper_pipe_l3769_376927

/-- The number of meters of copper pipe bought by the plumber -/
def copper_pipe : ℕ := sorry

/-- The number of meters of plastic pipe bought by the plumber -/
def plastic_pipe : ℕ := sorry

/-- The cost of one meter of pipe in dollars -/
def cost_per_meter : ℕ := 4

/-- The total cost of all pipes in dollars -/
def total_cost : ℕ := 100

theorem plumber_copper_pipe :
  copper_pipe = 10 ∧
  plastic_pipe = copper_pipe + 5 ∧
  cost_per_meter * (copper_pipe + plastic_pipe) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_plumber_copper_pipe_l3769_376927


namespace NUMINAMATH_CALUDE_solve_sandwich_problem_l3769_376967

/-- Represents the sandwich eating problem over two days -/
def sandwich_problem (total : ℕ) (first_day_fraction : ℚ) (remaining : ℕ) : Prop :=
  let first_day := (total : ℚ) * first_day_fraction
  let second_day := (total : ℕ) - first_day.floor - remaining
  first_day.floor - second_day = 2

/-- The theorem representing the sandwich problem -/
theorem solve_sandwich_problem :
  sandwich_problem 12 (1/2) 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_sandwich_problem_l3769_376967


namespace NUMINAMATH_CALUDE_carly_lollipop_ratio_l3769_376971

/-- Given a total number of lollipops and the number of grape lollipops,
    calculate the ratio of cherry lollipops to the total number of lollipops. -/
def lollipop_ratio (total : ℕ) (grape : ℕ) : ℚ :=
  let other_flavors := grape * 3
  let cherry := total - other_flavors
  (cherry : ℚ) / total

/-- Theorem stating that given the conditions in the problem,
    the ratio of cherry lollipops to the total is 1/2. -/
theorem carly_lollipop_ratio :
  lollipop_ratio 42 7 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_carly_lollipop_ratio_l3769_376971


namespace NUMINAMATH_CALUDE_unique_integer_sequence_l3769_376944

theorem unique_integer_sequence : ∃! x : ℤ, x = ((x + 2)/2 + 2)/2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_sequence_l3769_376944


namespace NUMINAMATH_CALUDE_system_solution_l3769_376952

theorem system_solution (x y : ℝ) : 
  (x^2 + 3*x*y = 12 ∧ x*y = 16 + y^2 - x*y - x^2) ↔ 
  ((x = 2 ∧ y = 1) ∨ (x = -2 ∧ y = -1)) :=
sorry

end NUMINAMATH_CALUDE_system_solution_l3769_376952


namespace NUMINAMATH_CALUDE_baker_cakes_sold_l3769_376984

/-- The number of cakes a baker intends to sell given certain pricing conditions -/
theorem baker_cakes_sold (n : ℝ) (h1 : n > 0) : ∃ x : ℕ,
  (n * x = 320) ∧
  (0.8 * n * (x + 2) = 320) ∧
  (x = 8) := by
sorry

end NUMINAMATH_CALUDE_baker_cakes_sold_l3769_376984


namespace NUMINAMATH_CALUDE_complex_square_norm_l3769_376963

theorem complex_square_norm (z : ℂ) (h : z^2 + Complex.normSq z = 8 - 3*I) : Complex.normSq z = 73/16 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_norm_l3769_376963


namespace NUMINAMATH_CALUDE_square_side_length_l3769_376933

theorem square_side_length 
  (total_wire : ℝ) 
  (triangle_perimeter : ℝ) 
  (h1 : total_wire = 78) 
  (h2 : triangle_perimeter = 46) : 
  (total_wire - triangle_perimeter) / 4 = 8 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l3769_376933


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3769_376916

/-- A geometric sequence of positive real numbers -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 3 + a 2 * a 4 + 2 * a 2 * a 3 = 49 →
  a 2 + a 3 = 7 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3769_376916


namespace NUMINAMATH_CALUDE_fruit_basket_cost_l3769_376985

/-- Represents the composition and prices of a fruit basket -/
structure FruitBasket where
  banana_count : ℕ
  apple_count : ℕ
  strawberry_count : ℕ
  avocado_count : ℕ
  banana_price : ℚ
  apple_price : ℚ
  strawberry_dozen_price : ℚ
  avocado_price : ℚ
  grape_half_bunch_price : ℚ

/-- Calculates the total cost of the fruit basket -/
def total_cost (fb : FruitBasket) : ℚ :=
  fb.banana_count * fb.banana_price +
  fb.apple_count * fb.apple_price +
  (fb.strawberry_count / 12) * fb.strawberry_dozen_price +
  fb.avocado_count * fb.avocado_price +
  2 * fb.grape_half_bunch_price

/-- The given fruit basket -/
def given_basket : FruitBasket := {
  banana_count := 4
  apple_count := 3
  strawberry_count := 24
  avocado_count := 2
  banana_price := 1
  apple_price := 2
  strawberry_dozen_price := 4
  avocado_price := 3
  grape_half_bunch_price := 2
}

/-- Theorem stating that the total cost of the given fruit basket is $28 -/
theorem fruit_basket_cost : total_cost given_basket = 28 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_cost_l3769_376985


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2903_l3769_376991

theorem smallest_prime_factor_of_2903 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2903 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2903 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2903_l3769_376991


namespace NUMINAMATH_CALUDE_pet_store_puppies_l3769_376918

theorem pet_store_puppies (sold : ℕ) (cages : ℕ) (puppies_per_cage : ℕ)
  (h1 : sold = 30)
  (h2 : cages = 6)
  (h3 : puppies_per_cage = 8) :
  sold + cages * puppies_per_cage = 78 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l3769_376918


namespace NUMINAMATH_CALUDE_only_negative_three_smaller_than_negative_two_l3769_376964

theorem only_negative_three_smaller_than_negative_two :
  (0 > -2) ∧ (-1 > -2) ∧ (-3 < -2) ∧ (1 > -2) :=
by sorry

end NUMINAMATH_CALUDE_only_negative_three_smaller_than_negative_two_l3769_376964


namespace NUMINAMATH_CALUDE_scenic_spot_selections_l3769_376975

theorem scenic_spot_selections (num_classes : ℕ) (num_spots : ℕ) : 
  num_classes = 3 → num_spots = 5 → (num_spots ^ num_classes) = 125 := by
  sorry

end NUMINAMATH_CALUDE_scenic_spot_selections_l3769_376975


namespace NUMINAMATH_CALUDE_sqrt_division_equality_l3769_376932

theorem sqrt_division_equality : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_division_equality_l3769_376932


namespace NUMINAMATH_CALUDE_train_length_l3769_376998

/-- Calculates the length of a train given its speed, the time it takes to cross a bridge, and the length of the bridge. -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 42 →
  crossing_time = 60 →
  bridge_length = 200 →
  ∃ (train_length : ℝ), abs (train_length - 500.2) < 0.1 :=
by
  sorry


end NUMINAMATH_CALUDE_train_length_l3769_376998


namespace NUMINAMATH_CALUDE_intersection_when_a_is_5_intersection_equals_A_iff_l3769_376955

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + (5-a)*x - 5*a ≤ 0}
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 6}

-- Define the complement of B
def complement_B : Set ℝ := {x | x < -3 ∨ 6 < x}

-- Theorem 1
theorem intersection_when_a_is_5 :
  A 5 ∩ complement_B = {x | -5 ≤ x ∧ x < -3} := by sorry

-- Theorem 2
theorem intersection_equals_A_iff (a : ℝ) :
  A a ∩ complement_B = A a ↔ a < -3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_5_intersection_equals_A_iff_l3769_376955


namespace NUMINAMATH_CALUDE_existence_of_roots_part_a_non_existence_of_roots_part_b_l3769_376936

-- Part a
theorem existence_of_roots_part_a : ∃ (a b : ℤ),
  (∀ x : ℝ, x^2 + a*x + b ≠ 0) ∧
  (∃ x : ℝ, ⌊x^2⌋ + a*x + b = 0) :=
sorry

-- Part b
theorem non_existence_of_roots_part_b : ¬∃ (a b : ℤ),
  (∀ x : ℝ, x^2 + 2*a*x + b ≠ 0) ∧
  (∃ x : ℝ, ⌊x^2⌋ + 2*a*x + b = 0) :=
sorry

end NUMINAMATH_CALUDE_existence_of_roots_part_a_non_existence_of_roots_part_b_l3769_376936


namespace NUMINAMATH_CALUDE_train_speed_crossing_bridge_l3769_376924

/-- The speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 130 →
  bridge_length = 150 →
  crossing_time = 15.506451791548985 →
  ∃ (speed : ℝ), (abs (speed - 64.9836) < 0.0001 ∧ 
    speed = (train_length + bridge_length) / crossing_time * 3.6) := by
  sorry

end NUMINAMATH_CALUDE_train_speed_crossing_bridge_l3769_376924


namespace NUMINAMATH_CALUDE_right_triangle_area_l3769_376941

theorem right_triangle_area (a b : ℝ) (ha : a = 45) (hb : b = 48) :
  (1 / 2 : ℝ) * a * b = 1080 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3769_376941


namespace NUMINAMATH_CALUDE_solve_for_D_l3769_376959

theorem solve_for_D : ∃ D : ℤ, 80 - (5 - (6 + 2 * (7 - 8 - D))) = 89 ∧ D = -5 := by sorry

end NUMINAMATH_CALUDE_solve_for_D_l3769_376959


namespace NUMINAMATH_CALUDE_simplify_like_terms_l3769_376962

theorem simplify_like_terms (y : ℝ) : 5*y + 2*y + 7*y = 14*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_like_terms_l3769_376962


namespace NUMINAMATH_CALUDE_jills_favorite_number_hint_l3769_376926

def is_even (n : Nat) : Prop := ∃ k, n = 2 * k

def has_repeating_prime_factors (n : Nat) : Prop :=
  ∃ p : Nat, Nat.Prime p ∧ (∃ k > 1, p ^ k ∣ n)

def best_guess : Nat := 84

theorem jills_favorite_number_hint :
  ∀ n : Nat,
  is_even n →
  has_repeating_prime_factors n →
  best_guess = 84 →
  (∃ p : Nat, Nat.Prime p ∧ (∃ k > 1, p ^ k ∣ n) ∧ p = 2) :=
by sorry

end NUMINAMATH_CALUDE_jills_favorite_number_hint_l3769_376926


namespace NUMINAMATH_CALUDE_fencing_requirement_l3769_376953

/-- Represents a rectangular field with given dimensions and fencing requirements. -/
structure RectangularField where
  length : ℝ
  width : ℝ
  area : ℝ
  uncovered_side : ℝ

/-- Calculates the required fencing for a rectangular field. -/
def required_fencing (field : RectangularField) : ℝ :=
  field.length + 2 * field.width

/-- Theorem stating the required fencing for a specific rectangular field. -/
theorem fencing_requirement (field : RectangularField)
  (h1 : field.area = 650)
  (h2 : field.uncovered_side = 20)
  (h3 : field.area = field.length * field.width)
  (h4 : field.length = field.uncovered_side) :
  required_fencing field = 85 :=
sorry

end NUMINAMATH_CALUDE_fencing_requirement_l3769_376953


namespace NUMINAMATH_CALUDE_scientists_sum_equals_total_germany_japan_us_ratio_l3769_376969

/-- The total number of scientists in the research project. -/
def total_scientists : ℕ := 150

/-- The number of scientists from Germany. -/
def germany_scientists : ℕ := 27

/-- The number of scientists from other European countries. -/
def other_europe_scientists : ℕ := 33

/-- The number of scientists from Japan. -/
def japan_scientists : ℕ := 18

/-- The number of scientists from China. -/
def china_scientists : ℕ := 15

/-- The number of scientists from other Asian countries. -/
def other_asia_scientists : ℕ := 12

/-- The number of scientists from Canada. -/
def canada_scientists : ℕ := 23

/-- The number of scientists from the United States. -/
def us_scientists : ℕ := 12

/-- The number of scientists from South America. -/
def south_america_scientists : ℕ := 8

/-- The number of scientists from Australia. -/
def australia_scientists : ℕ := 3

/-- Theorem stating that the sum of scientists from all countries equals the total number of scientists. -/
theorem scientists_sum_equals_total :
  germany_scientists + other_europe_scientists + japan_scientists + china_scientists +
  other_asia_scientists + canada_scientists + us_scientists + south_america_scientists +
  australia_scientists = total_scientists :=
by sorry

/-- Theorem stating the ratio of scientists from Germany, Japan, and the United States. -/
theorem germany_japan_us_ratio :
  ∃ (k : ℕ), k ≠ 0 ∧ germany_scientists = 9 * k ∧ japan_scientists = 6 * k ∧ us_scientists = 4 * k :=
by sorry

end NUMINAMATH_CALUDE_scientists_sum_equals_total_germany_japan_us_ratio_l3769_376969


namespace NUMINAMATH_CALUDE_dinner_cakes_count_l3769_376981

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := 5

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := 3

/-- The total number of cakes served over two days -/
def total_cakes : ℕ := 14

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := total_cakes - lunch_cakes - yesterday_cakes

theorem dinner_cakes_count : dinner_cakes = 6 := by sorry

end NUMINAMATH_CALUDE_dinner_cakes_count_l3769_376981


namespace NUMINAMATH_CALUDE_count_divisible_by_33_l3769_376979

/-- Represents a 10-digit number of the form a2016b2017 -/
def NumberForm (a b : Nat) : Nat :=
  a * 10^9 + 2 * 10^8 + 0 * 10^7 + 1 * 10^6 + 6 * 10^5 + b * 10^4 + 2 * 10^3 + 0 * 10^2 + 1 * 10 + 7

/-- Predicate to check if a number is a single digit -/
def IsSingleDigit (n : Nat) : Prop := n < 10

/-- Main theorem -/
theorem count_divisible_by_33 :
  ∃! (count : Nat), ∃ (S : Finset (Nat × Nat)),
    (∀ (pair : Nat × Nat), pair ∈ S ↔ 
      IsSingleDigit pair.1 ∧ 
      IsSingleDigit pair.2 ∧ 
      (NumberForm pair.1 pair.2) % 33 = 0) ∧
    S.card = count ∧
    count = 3 := by sorry

end NUMINAMATH_CALUDE_count_divisible_by_33_l3769_376979


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3769_376997

theorem imaginary_part_of_complex_number (z : ℂ) :
  (z.re > 0) →
  (z.im = 2 * z.re) →
  (Complex.abs z = Real.sqrt 5) →
  z.im = 2 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3769_376997


namespace NUMINAMATH_CALUDE_min_value_theorem_l3769_376912

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  x + 3 * y ≥ 4 + 8 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧
    x₀ + 3 * y₀ = 4 + 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3769_376912


namespace NUMINAMATH_CALUDE_police_emergency_number_has_large_prime_divisor_l3769_376903

/-- A police emergency number is a positive integer that ends with 133 in decimal notation. -/
def is_police_emergency_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 1000 * k + 133

/-- Every police emergency number has a prime divisor greater than 7. -/
theorem police_emergency_number_has_large_prime_divisor (n : ℕ) 
  (h : is_police_emergency_number n) : 
  ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n := by
sorry

end NUMINAMATH_CALUDE_police_emergency_number_has_large_prime_divisor_l3769_376903


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3769_376956

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3769_376956


namespace NUMINAMATH_CALUDE_solution_x_fourth_plus_81_l3769_376908

theorem solution_x_fourth_plus_81 :
  let solutions : List ℂ := [
    Complex.mk ((3 * Real.sqrt 2) / 2) ((3 * Real.sqrt 2) / 2),
    Complex.mk (-(3 * Real.sqrt 2) / 2) (-(3 * Real.sqrt 2) / 2),
    Complex.mk (-(3 * Real.sqrt 2) / 2) ((3 * Real.sqrt 2) / 2),
    Complex.mk ((3 * Real.sqrt 2) / 2) (-(3 * Real.sqrt 2) / 2)
  ]
  ∀ z : ℂ, z^4 + 81 = 0 ↔ z ∈ solutions := by
sorry

end NUMINAMATH_CALUDE_solution_x_fourth_plus_81_l3769_376908


namespace NUMINAMATH_CALUDE_stove_and_wall_repair_cost_l3769_376900

/-- The total cost of replacing a stove and repairing wall damage -/
theorem stove_and_wall_repair_cost :
  let stove_cost : ℚ := 1200
  let wall_repair_cost : ℚ := stove_cost / 6
  let total_cost : ℚ := stove_cost + wall_repair_cost
  total_cost = 1400 := by sorry

end NUMINAMATH_CALUDE_stove_and_wall_repair_cost_l3769_376900


namespace NUMINAMATH_CALUDE_probability_third_key_opens_door_l3769_376911

/-- The probability of opening a door with the third key, given 5 keys with only one correct key --/
theorem probability_third_key_opens_door : 
  ∀ (n : ℕ) (p : ℝ),
    n = 5 →  -- There are 5 keys
    p = 1 / n →  -- The probability of selecting the correct key is 1/n
    p = 1 / 5  -- The probability of opening the door on the third attempt is 1/5
    := by sorry

end NUMINAMATH_CALUDE_probability_third_key_opens_door_l3769_376911


namespace NUMINAMATH_CALUDE_exponent_expression_equality_l3769_376968

theorem exponent_expression_equality : 3^(1^(0^8)) + ((3^1)^0)^8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_expression_equality_l3769_376968


namespace NUMINAMATH_CALUDE_rachels_age_problem_l3769_376905

/-- Rachel's age problem -/
theorem rachels_age_problem (rachel_age : ℕ) (grandfather_age : ℕ) (mother_age : ℕ) (father_age : ℕ) :
  rachel_age = 12 →
  grandfather_age = 7 * rachel_age →
  father_age = mother_age + 5 →
  father_age + (25 - rachel_age) = 60 →
  mother_age / grandfather_age = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rachels_age_problem_l3769_376905


namespace NUMINAMATH_CALUDE_unique_triple_l3769_376965

theorem unique_triple : ∀ a b c : ℕ+,
  a ≤ b → b ≤ c →
  Nat.gcd (a.val) (Nat.gcd (b.val) (c.val)) = 1 →
  (a.val^3 + b.val^3 + c.val^3) % (a.val^2 * b.val) = 0 →
  (a.val^3 + b.val^3 + c.val^3) % (b.val^2 * c.val) = 0 →
  (a.val^3 + b.val^3 + c.val^3) % (c.val^2 * a.val) = 0 →
  a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_l3769_376965


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l3769_376996

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := (1 - i) * z = 2 + 3 * i

-- Define the second quadrant
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem z_in_second_quadrant :
  ∃ z : ℂ, equation z ∧ second_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l3769_376996


namespace NUMINAMATH_CALUDE_chord_length_is_sqrt_6_l3769_376957

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x + y^2 + 4*x - 4*y + 6 = 0

-- Define the line l
def line_l (k x y : ℝ) : Prop := k*x + y + 4 = 0

-- Define the line m
def line_m (k x y : ℝ) : Prop := y = x + k

-- Theorem statement
theorem chord_length_is_sqrt_6 (k : ℝ) :
  (∃ x y : ℝ, line_l k x y ∧ circle_C x y) →  -- l is a symmetric axis of C
  (∃ x y : ℝ, line_m k x y ∧ circle_C x y) →  -- m intersects C
  (∃ x1 y1 x2 y2 : ℝ, 
    line_m k x1 y1 ∧ circle_C x1 y1 ∧
    line_m k x2 y2 ∧ circle_C x2 y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_is_sqrt_6_l3769_376957


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l3769_376929

def ring_arrangements (total_rings : ℕ) (chosen_rings : ℕ) (fingers : ℕ) : ℕ :=
  (total_rings.choose chosen_rings) * (chosen_rings.factorial) * ((chosen_rings + fingers - 1).choose (fingers - 1))

theorem ring_arrangement_count :
  ring_arrangements 8 5 4 = 376320 :=
by sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l3769_376929


namespace NUMINAMATH_CALUDE_evaluate_expression_l3769_376937

theorem evaluate_expression (x y : ℝ) (hx : x = 4) (hy : y = 9) :
  2 * x^(y/2) + 5 * y^(x/2) = 1429 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3769_376937


namespace NUMINAMATH_CALUDE_trig_identity_l3769_376930

theorem trig_identity :
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (47 * π / 180) * Real.cos (103 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3769_376930


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l3769_376983

-- Define the conditions
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  ∃ d : ℝ, b - a = c - b ∧ b - a = d ∧ d ≠ 0

def is_geometric_sequence (c a b : ℝ) : Prop :=
  ∃ r : ℝ, a / c = b / a ∧ a / c = r ∧ r ≠ 1

-- State the theorem
theorem arithmetic_geometric_sequence_sum (a b c : ℝ) :
  a ≠ b ∧ b ≠ c ∧ c ≠ a →
  is_arithmetic_sequence a b c →
  is_geometric_sequence c a b →
  a + 3*b + c = 10 →
  a = -4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l3769_376983


namespace NUMINAMATH_CALUDE_nell_gave_28_cards_l3769_376919

/-- The number of cards Nell gave to Jeff -/
def cards_given_to_jeff (initial_cards : ℕ) (remaining_cards : ℕ) : ℕ :=
  initial_cards - remaining_cards

/-- Proof that Nell gave 28 cards to Jeff -/
theorem nell_gave_28_cards :
  cards_given_to_jeff 304 276 = 28 := by
  sorry

end NUMINAMATH_CALUDE_nell_gave_28_cards_l3769_376919


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l3769_376922

theorem quadratic_real_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 2 * x^2 - 3 * x + m = 0 ∧ 2 * y^2 - 3 * y + m = 0) ↔ m ≤ 9/8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l3769_376922


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3769_376961

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - I) = abs (1 - I) + I) :
  z.im = (Real.sqrt 2 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3769_376961


namespace NUMINAMATH_CALUDE_remaining_volume_of_cone_volume_of_remaining_part_is_27_l3769_376925

/-- Represents a cone with an inscribed sphere and a plane through the circle of tangency -/
structure InscribedSphereCone where
  /-- The angle between the slant height and the base plane -/
  α : Real
  /-- The volume of the part of the cone enclosed between the tangency plane and the base plane -/
  enclosed_volume : Real

/-- Theorem stating the volume of the remaining part of the cone -/
theorem remaining_volume_of_cone (cone : InscribedSphereCone) 
  (h1 : cone.α = Real.arccos (1/4))
  (h2 : cone.enclosed_volume = 37) :
  64 - cone.enclosed_volume = 27 := by
  sorry

/-- Main theorem to prove -/
theorem volume_of_remaining_part_is_27 (cone : InscribedSphereCone) 
  (h1 : cone.α = Real.arccos (1/4))
  (h2 : cone.enclosed_volume = 37) :
  ∃ (v : Real), v = 27 ∧ v = 64 - cone.enclosed_volume := by
  sorry

end NUMINAMATH_CALUDE_remaining_volume_of_cone_volume_of_remaining_part_is_27_l3769_376925


namespace NUMINAMATH_CALUDE_five_b_value_l3769_376993

theorem five_b_value (a b : ℚ) (h1 : 3 * a + 4 * b = 4) (h2 : b - 3 = a) : 5 * b = 65 / 7 := by
  sorry

end NUMINAMATH_CALUDE_five_b_value_l3769_376993


namespace NUMINAMATH_CALUDE_jessie_min_score_l3769_376939

/-- Represents the test scores and conditions for Jessie's problem -/
structure TestScores where
  max_score : ℕ
  first_three : Fin 3 → ℕ
  total_tests : ℕ
  target_average : ℕ

/-- The minimum score needed on one of the remaining tests -/
def min_score (ts : TestScores) : ℕ :=
  let total_needed := ts.target_average * ts.total_tests
  let current_total := (ts.first_three 0) + (ts.first_three 1) + (ts.first_three 2)
  let remaining_total := total_needed - current_total
  remaining_total - 2 * ts.max_score

/-- Theorem stating the minimum score Jessie needs to achieve -/
theorem jessie_min_score :
  let ts : TestScores := {
    max_score := 120,
    first_three := ![88, 105, 96],
    total_tests := 6,
    target_average := 90
  }
  min_score ts = 11 := by sorry

end NUMINAMATH_CALUDE_jessie_min_score_l3769_376939


namespace NUMINAMATH_CALUDE_office_printer_paper_duration_l3769_376978

/-- The number of days printer paper will last given the number of packs, sheets per pack, and daily usage. -/
def printer_paper_duration (packs : ℕ) (sheets_per_pack : ℕ) (daily_usage : ℕ) : ℕ :=
  (packs * sheets_per_pack) / daily_usage

/-- Theorem stating that under the given conditions, the printer paper will last 6 days. -/
theorem office_printer_paper_duration :
  let packs : ℕ := 2
  let sheets_per_pack : ℕ := 240
  let daily_usage : ℕ := 80
  printer_paper_duration packs sheets_per_pack daily_usage = 6 := by
  sorry


end NUMINAMATH_CALUDE_office_printer_paper_duration_l3769_376978


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l3769_376945

-- Define the set of real numbers except 1
def RealExceptOne : Set ℝ := {x : ℝ | x ≠ 1}

-- Define the property of the expression being meaningful
def IsMeaningful (x : ℝ) : Prop := x - 1 ≠ 0

-- Theorem statement
theorem meaningful_expression_range :
  ∀ x : ℝ, IsMeaningful x ↔ x ∈ RealExceptOne :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l3769_376945


namespace NUMINAMATH_CALUDE_q_undetermined_l3769_376977

theorem q_undetermined (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  (q ∨ ¬q) := by sorry

end NUMINAMATH_CALUDE_q_undetermined_l3769_376977


namespace NUMINAMATH_CALUDE_geometric_sequence_q_eq_one_l3769_376989

/-- A positive geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ a (n + 1) = q * a n

theorem geometric_sequence_q_eq_one
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_prod : a 2 * a 6 = 16)
  (h_sum : a 4 + a 8 = 8) :
  q = 1 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_q_eq_one_l3769_376989


namespace NUMINAMATH_CALUDE_cyclist_average_speed_l3769_376982

/-- The average speed of a cyclist driving four laps of equal distance at different speeds -/
theorem cyclist_average_speed (d : ℝ) (h : d > 0) : 
  let speeds := [6, 12, 18, 24]
  let total_distance := 4 * d
  let total_time := (d / 6 + d / 12 + d / 18 + d / 24)
  total_distance / total_time = 288 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_average_speed_l3769_376982


namespace NUMINAMATH_CALUDE_room_population_l3769_376910

theorem room_population (P : ℕ) 
  (women_ratio : (2 : ℚ) / 5 * P = (P : ℚ).floor)
  (married_ratio : (1 : ℚ) / 2 * P = (P : ℚ).floor)
  (max_unmarried_women : ℕ → Prop)
  (h_max_unmarried : max_unmarried_women 32) :
  P = 64 := by
sorry

end NUMINAMATH_CALUDE_room_population_l3769_376910


namespace NUMINAMATH_CALUDE_percentage_of_prize_money_kept_l3769_376947

-- Define the original repair cost
def original_repair_cost : ℝ := 20000

-- Define the discount percentage
def discount_percentage : ℝ := 0.20

-- Define the prize money
def prize_money : ℝ := 70000

-- Define John's profit
def profit : ℝ := 47000

-- Theorem to prove
theorem percentage_of_prize_money_kept (ε : ℝ) (h : ε > 0) :
  ∃ (percentage : ℝ), 
    abs (percentage - (profit / prize_money * 100)) < ε ∧ 
    abs (percentage - 67.14) < ε :=
sorry

end NUMINAMATH_CALUDE_percentage_of_prize_money_kept_l3769_376947


namespace NUMINAMATH_CALUDE_irrational_floor_bijection_l3769_376938

theorem irrational_floor_bijection (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hirra : Irrational a) (hirrb : Irrational b) 
  (hab : 1 / a + 1 / b = 1) :
  ∀ k : ℕ, ∃! (m n : ℕ), (⌊m * a⌋ = k ∨ ⌊n * b⌋ = k) ∧
    ∀ (p q : ℕ), (⌊p * a⌋ = k ∨ ⌊q * b⌋ = k) → (p = m ∧ ⌊p * a⌋ = k) ∨ (q = n ∧ ⌊q * b⌋ = k) :=
by sorry

end NUMINAMATH_CALUDE_irrational_floor_bijection_l3769_376938


namespace NUMINAMATH_CALUDE_problem_solution_l3769_376942

theorem problem_solution : 
  ∀ x y : ℤ, x > y ∧ y > 0 ∧ x + y + x * y = 152 → x = 16 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3769_376942


namespace NUMINAMATH_CALUDE_sixth_group_frequency_is_one_tenth_l3769_376946

/-- Represents the distribution of students across six groups in a mathematics competition. -/
structure StudentDistribution where
  total : ℕ
  group1 : ℕ
  group2 : ℕ
  group3 : ℕ
  group4 : ℕ
  freq5 : ℚ

/-- Calculates the frequency of the sixth group given a student distribution. -/
def sixthGroupFrequency (d : StudentDistribution) : ℚ :=
  1 - (d.group1 + d.group2 + d.group3 + d.group4 : ℚ) / d.total - d.freq5

/-- Theorem stating that for the given distribution, the frequency of the sixth group is 0.1. -/
theorem sixth_group_frequency_is_one_tenth 
  (d : StudentDistribution)
  (h1 : d.total = 40)
  (h2 : d.group1 = 10)
  (h3 : d.group2 = 5)
  (h4 : d.group3 = 7)
  (h5 : d.group4 = 6)
  (h6 : d.freq5 = 1/5) :
  sixthGroupFrequency d = 1/10 := by
  sorry


end NUMINAMATH_CALUDE_sixth_group_frequency_is_one_tenth_l3769_376946


namespace NUMINAMATH_CALUDE_vertex_of_f_l3769_376995

/-- The quadratic function f(x) = -3(x+1)^2 - 2 -/
def f (x : ℝ) : ℝ := -3 * (x + 1)^2 - 2

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (-1, -2)

/-- Theorem: The vertex of the quadratic function f is (-1, -2) -/
theorem vertex_of_f : 
  (∀ x : ℝ, f x ≤ f (vertex.1)) ∧ f (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_vertex_of_f_l3769_376995


namespace NUMINAMATH_CALUDE_cost_per_share_is_50_l3769_376960

/-- Represents the savings and investment scenario of a married couple --/
structure SavingsScenario where
  wife_weekly_savings : ℕ
  husband_monthly_savings : ℕ
  savings_period_months : ℕ
  investment_fraction : ℚ
  num_shares_bought : ℕ

/-- Calculates the cost per share of stock based on the given savings scenario --/
def cost_per_share (scenario : SavingsScenario) : ℚ :=
  let total_savings := (scenario.wife_weekly_savings * 4 * scenario.savings_period_months +
                        scenario.husband_monthly_savings * scenario.savings_period_months)
  let investment_amount := (total_savings : ℚ) * scenario.investment_fraction
  investment_amount / scenario.num_shares_bought

/-- Theorem stating that the cost per share is $50 for the given scenario --/
theorem cost_per_share_is_50 (scenario : SavingsScenario)
  (h1 : scenario.wife_weekly_savings = 100)
  (h2 : scenario.husband_monthly_savings = 225)
  (h3 : scenario.savings_period_months = 4)
  (h4 : scenario.investment_fraction = 1/2)
  (h5 : scenario.num_shares_bought = 25) :
  cost_per_share scenario = 50 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_share_is_50_l3769_376960


namespace NUMINAMATH_CALUDE_simplify_expression_l3769_376966

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^3 - b^3) / (a * b^2) - (a * b^2 - b^3) / (a * b^2 - a^3) = (a^3 - a * b^2 + b^4) / (a * b^2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3769_376966


namespace NUMINAMATH_CALUDE_heart_five_three_l3769_376921

-- Define the ♥ operation
def heart (x y : ℝ) : ℝ := 4 * x - 2 * y

-- Theorem statement
theorem heart_five_three : heart 5 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_heart_five_three_l3769_376921


namespace NUMINAMATH_CALUDE_trapezium_area_l3769_376935

-- Define the trapezium properties
def a : ℝ := 10 -- Length of one parallel side
def b : ℝ := 18 -- Length of the other parallel side
def h : ℝ := 15 -- Distance between parallel sides

-- Theorem statement
theorem trapezium_area : (1/2 : ℝ) * (a + b) * h = 210 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_area_l3769_376935


namespace NUMINAMATH_CALUDE_win_rate_problem_l3769_376928

/-- Represents the win rate problem for a sports team -/
theorem win_rate_problem (first_third_win_rate : ℚ) (total_matches : ℕ) :
  first_third_win_rate = 55 / 100 →
  (∃ (remaining_win_rate : ℚ),
    remaining_win_rate = 85 / 100 ∧
    first_third_win_rate * (1 / 3) + remaining_win_rate * (2 / 3) = 3 / 4) ∧
  (first_third_win_rate * (1 / 3) + 1 * (2 / 3) = 85 / 100) :=
by sorry

end NUMINAMATH_CALUDE_win_rate_problem_l3769_376928


namespace NUMINAMATH_CALUDE_four_digit_integer_problem_l3769_376909

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def middle_digits_sum (n : ℕ) : ℕ := ((n / 100) % 10) + ((n / 10) % 10)

def thousands_minus_units (n : ℕ) : ℤ := (n / 1000 : ℤ) - (n % 10 : ℤ)

theorem four_digit_integer_problem (n : ℕ) 
  (h1 : is_four_digit n)
  (h2 : digit_sum n = 16)
  (h3 : middle_digits_sum n = 10)
  (h4 : thousands_minus_units n = 2)
  (h5 : n % 9 = 0) :
  n = 4522 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_integer_problem_l3769_376909


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3769_376920

theorem inequality_solution_set (a : ℝ) (h : a > 0) :
  let solution_set := {x : ℝ | a * (x - 1) / (x - 2) > 1}
  (a = 1 → solution_set = {x : ℝ | x > 2}) ∧
  (0 < a ∧ a < 1 → solution_set = {x : ℝ | (a - 2) / (1 - a) < x ∧ x < 2}) ∧
  (a > 1 → solution_set = {x : ℝ | x < (a - 2) / (a - 1) ∨ x > 2}) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3769_376920


namespace NUMINAMATH_CALUDE_tan_405_degrees_l3769_376907

theorem tan_405_degrees : Real.tan (405 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_405_degrees_l3769_376907
