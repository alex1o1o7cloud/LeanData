import Mathlib

namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l1439_143902

theorem gcd_lcm_sum : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l1439_143902


namespace NUMINAMATH_CALUDE_triangle_theorem_l1439_143962

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 2 ∧
  (Real.cos t.C) * t.a * t.b = 1 ∧
  1/2 * t.a * t.b * (Real.sin t.C) = 1/2 ∧
  (Real.sin t.A) * (Real.cos t.A) = Real.sqrt 3 / 4

-- State the theorem
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.C = Real.pi / 4 ∧ (t.c = Real.sqrt 6 ∨ t.c = 2 * Real.sqrt 6 / 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1439_143962


namespace NUMINAMATH_CALUDE_balls_triangle_to_square_l1439_143946

theorem balls_triangle_to_square (n : ℕ) (h1 : n * (n + 1) / 2 = 1176) :
  let square_side := n - 8
  square_side * square_side - n * (n + 1) / 2 = 424 := by
  sorry

end NUMINAMATH_CALUDE_balls_triangle_to_square_l1439_143946


namespace NUMINAMATH_CALUDE_abs_difference_equals_seven_l1439_143908

theorem abs_difference_equals_seven (a b : ℝ) 
  (ha : |a| = 4) 
  (hb : |b| = 3) 
  (hab : a * b < 0) : 
  |a - b| = 7 := by
sorry

end NUMINAMATH_CALUDE_abs_difference_equals_seven_l1439_143908


namespace NUMINAMATH_CALUDE_math_club_team_probability_l1439_143959

theorem math_club_team_probability :
  let total_girls : ℕ := 8
  let total_boys : ℕ := 6
  let team_size : ℕ := 4
  let girls_in_team : ℕ := 2
  let boys_in_team : ℕ := 2

  (Nat.choose total_girls girls_in_team * Nat.choose total_boys boys_in_team) /
  Nat.choose (total_girls + total_boys) team_size = 60 / 143 :=
by sorry

end NUMINAMATH_CALUDE_math_club_team_probability_l1439_143959


namespace NUMINAMATH_CALUDE_paper_folding_thickness_l1439_143987

theorem paper_folding_thickness (initial_thickness : ℝ) (target_thickness : ℝ) : 
  initial_thickness > 0 → target_thickness > 0 →
  (∃ n : ℕ, (2^n : ℝ) * initial_thickness > target_thickness) →
  (∀ m : ℕ, m < 8 → (2^m : ℝ) * initial_thickness ≤ target_thickness) →
  (∃ n : ℕ, n = 8 ∧ (2^n : ℝ) * initial_thickness > target_thickness) :=
by sorry

end NUMINAMATH_CALUDE_paper_folding_thickness_l1439_143987


namespace NUMINAMATH_CALUDE_triangle_sides_l1439_143949

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (3 * Real.pi + x) * Real.cos (Real.pi - x) + (Real.cos (Real.pi / 2 + x))^2

theorem triangle_sides (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi →
  f A = 3/2 →
  a = 2 →
  b + c = 4 →
  b = 2 ∧ c = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_sides_l1439_143949


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l1439_143995

-- Define the variables
variable (a b c d x y z : ℝ)

-- Define the given ratios
def ratio_a_to_b : a / b = 2 * x / (3 * y) := by sorry
def ratio_b_to_c : b / c = z / (5 * z) := by sorry
def ratio_a_to_d : a / d = 4 * x / (7 * y) := by sorry
def ratio_d_to_c : d / c = 7 * y / (3 * z) := by sorry

-- State the theorem
theorem ratio_a_to_c (ha : a > 0) (hc : c > 0) : a / c = 2 * x / (15 * y) := by sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l1439_143995


namespace NUMINAMATH_CALUDE_unique_solution_inequality_l1439_143994

theorem unique_solution_inequality (x : ℝ) :
  (x > 0 ∧ x * Real.sqrt (18 - x) + Real.sqrt (24 * x - x^3) ≥ 18) ↔ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_l1439_143994


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1439_143954

/-- A quadratic function f(x) = ax^2 + bx + c where the solution set of ax^2 + bx + c > 0 is {x | -1 < x < 3} -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The solution set condition -/
def SolutionSetCondition (a b c : ℝ) : Prop :=
  ∀ x, QuadraticFunction a b c x > 0 ↔ -1 < x ∧ x < 3

theorem quadratic_inequality (a b c : ℝ) (h : SolutionSetCondition a b c) :
  QuadraticFunction a b c 5 < QuadraticFunction a b c (-1) ∧
  QuadraticFunction a b c (-1) < QuadraticFunction a b c 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1439_143954


namespace NUMINAMATH_CALUDE_reflect_A_across_y_axis_l1439_143973

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- The original point A -/
def A : ℝ × ℝ := (2, -1)

theorem reflect_A_across_y_axis :
  reflect_y A = (-2, -1) := by sorry

end NUMINAMATH_CALUDE_reflect_A_across_y_axis_l1439_143973


namespace NUMINAMATH_CALUDE_school_gender_ratio_l1439_143971

/-- Given a school with a 5:4 ratio of boys to girls and 1500 boys, prove there are 1200 girls. -/
theorem school_gender_ratio (num_boys : ℕ) (num_girls : ℕ) : 
  (num_boys : ℚ) / num_girls = 5 / 4 →
  num_boys = 1500 →
  num_girls = 1200 := by
  sorry

end NUMINAMATH_CALUDE_school_gender_ratio_l1439_143971


namespace NUMINAMATH_CALUDE_smallest_positive_solution_of_quartic_l1439_143944

theorem smallest_positive_solution_of_quartic (x : ℝ) :
  x^4 - 50*x^2 + 576 = 0 ∧ x > 0 → x = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_of_quartic_l1439_143944


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1439_143934

def A : Set ℝ := {Real.sin (90 * Real.pi / 180), Real.cos (180 * Real.pi / 180)}
def B : Set ℝ := {x : ℝ | x^2 + x = 0}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1439_143934


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l1439_143992

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_slopes_equal {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of k for which the lines y = 5x + 3 and y = (3k)x + 1 are parallel -/
theorem parallel_lines_k_value :
  (∀ x y : ℝ, y = 5 * x + 3 ↔ y = (3 * k) * x + 1) → k = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l1439_143992


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l1439_143974

/-- The line 5x + 12y + a = 0 is tangent to the circle (x-1)^2 + y^2 = 1 if and only if a = 8 or a = -18 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, (5 * x + 12 * y + a = 0) → ((x - 1)^2 + y^2 = 1)) ↔ (a = 8 ∨ a = -18) := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l1439_143974


namespace NUMINAMATH_CALUDE_unique_triple_l1439_143964

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a * b * c < a + b + c ∧
  (a + b + c = 6)

theorem unique_triple : ∀ a b c : ℕ,
  is_valid_triple a b c → (a = 1 ∧ b = 1 ∧ c = 4) ∨ (a = 1 ∧ b = 4 ∧ c = 1) ∨ (a = 4 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_l1439_143964


namespace NUMINAMATH_CALUDE_park_trees_l1439_143930

theorem park_trees (willows : ℕ) (oaks : ℕ) : 
  willows = 36 → oaks = willows + 11 → willows + oaks = 83 := by
  sorry

end NUMINAMATH_CALUDE_park_trees_l1439_143930


namespace NUMINAMATH_CALUDE_metal_sheet_width_l1439_143933

/-- Represents the dimensions and volume of a box created from a metal sheet -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ

/-- Calculates the original width of the metal sheet given the box dimensions -/
def calculate_original_width (box : BoxDimensions) : ℝ :=
  box.width + 2 * box.height

/-- Theorem stating that given the specified conditions, the original width of the sheet must be 36 m -/
theorem metal_sheet_width
  (box : BoxDimensions)
  (h1 : box.length = 48 - 2 * 4)
  (h2 : box.height = 4)
  (h3 : box.volume = 4480)
  (h4 : box.volume = box.length * box.width * box.height) :
  calculate_original_width box = 36 := by
  sorry

#check metal_sheet_width

end NUMINAMATH_CALUDE_metal_sheet_width_l1439_143933


namespace NUMINAMATH_CALUDE_min_value_quadratic_plus_constant_l1439_143986

theorem min_value_quadratic_plus_constant :
  (∀ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 ≥ 2018) ∧
  (∃ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 = 2018) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_plus_constant_l1439_143986


namespace NUMINAMATH_CALUDE_john_volunteer_frequency_l1439_143967

/-- The number of hours John volunteers per year -/
def annual_hours : ℕ := 72

/-- The number of hours per volunteering session -/
def hours_per_session : ℕ := 3

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- The number of times John volunteers per month -/
def volunteer_times_per_month : ℚ :=
  (annual_hours / hours_per_session : ℚ) / months_per_year

theorem john_volunteer_frequency :
  volunteer_times_per_month = 2 := by
  sorry

end NUMINAMATH_CALUDE_john_volunteer_frequency_l1439_143967


namespace NUMINAMATH_CALUDE_expansion_terms_imply_n_equals_10_l1439_143918

theorem expansion_terms_imply_n_equals_10 (x a : ℝ) (n : ℕ) :
  (n.choose 1 : ℝ) * x^(n - 1) * a = 210 →
  (n.choose 2 : ℝ) * x^(n - 2) * a^2 = 840 →
  (n.choose 3 : ℝ) * x^(n - 3) * a^3 = 2520 →
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_expansion_terms_imply_n_equals_10_l1439_143918


namespace NUMINAMATH_CALUDE_other_communities_count_l1439_143957

theorem other_communities_count (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ) 
  (h_total : total = 400)
  (h_muslim : muslim_percent = 44/100)
  (h_hindu : hindu_percent = 28/100)
  (h_sikh : sikh_percent = 10/100) :
  (total : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 72 := by
  sorry

end NUMINAMATH_CALUDE_other_communities_count_l1439_143957


namespace NUMINAMATH_CALUDE_eraser_price_is_75_cents_l1439_143996

/-- The price of each eraser sold by the student council -/
def price_per_eraser (num_boxes : ℕ) (erasers_per_box : ℕ) (total_revenue : ℚ) : ℚ :=
  total_revenue / (num_boxes * erasers_per_box)

/-- Theorem: The price of each eraser is $0.75 -/
theorem eraser_price_is_75_cents :
  price_per_eraser 48 24 864 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_eraser_price_is_75_cents_l1439_143996


namespace NUMINAMATH_CALUDE_purchase_price_calculation_l1439_143920

/-- The purchase price of an article given markup conditions -/
theorem purchase_price_calculation (M : ℝ) (P : ℝ) : 
  M = 0.30 * P + 12 → M = 55 → P = 143.33 := by sorry

end NUMINAMATH_CALUDE_purchase_price_calculation_l1439_143920


namespace NUMINAMATH_CALUDE_rationality_of_expressions_l1439_143935

theorem rationality_of_expressions :
  (∃ (a b : ℤ), b ≠ 0 ∧ (1.728 : ℚ) = a / b) ∧
  (∃ (c d : ℤ), d ≠ 0 ∧ (0.0032 : ℚ) = c / d) ∧
  (∃ (e f : ℤ), f ≠ 0 ∧ (-8 : ℚ) = e / f) ∧
  (∃ (g h : ℤ), h ≠ 0 ∧ (0.25 : ℚ) = g / h) ∧
  ¬(∃ (i j : ℤ), j ≠ 0 ∧ Real.pi = (i : ℚ) / j) :=
by sorry

end NUMINAMATH_CALUDE_rationality_of_expressions_l1439_143935


namespace NUMINAMATH_CALUDE_inequality_never_satisfied_l1439_143976

theorem inequality_never_satisfied (m : ℝ) :
  (∀ x : ℝ, ¬(|x - 4| + |3 - x| < m)) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_never_satisfied_l1439_143976


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1439_143924

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b ∧ b > 1 → log a 3 < log b 3) ∧
  (∃ a b, log a 3 < log b 3 ∧ ¬(a > b ∧ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1439_143924


namespace NUMINAMATH_CALUDE_max_b_value_max_b_value_achieved_l1439_143958

theorem max_b_value (x b : ℤ) (h1 : x^2 + b*x = -21) (h2 : b > 0) : b ≤ 22 := by
  sorry

theorem max_b_value_achieved : ∃ x b : ℤ, x^2 + b*x = -21 ∧ b > 0 ∧ b = 22 := by
  sorry

end NUMINAMATH_CALUDE_max_b_value_max_b_value_achieved_l1439_143958


namespace NUMINAMATH_CALUDE_circle_condition_l1439_143912

theorem circle_condition (m : ℝ) : 
  (∃ (a b r : ℝ), ∀ (x y : ℝ), (x^2 + y^2 - x + y + m = 0) ↔ ((x - a)^2 + (y - b)^2 = r^2)) → 
  m < 1/2 := by
sorry

end NUMINAMATH_CALUDE_circle_condition_l1439_143912


namespace NUMINAMATH_CALUDE_double_round_robin_max_teams_l1439_143956

/-- The maximum number of teams in a double round-robin tournament --/
def max_teams : ℕ := 6

/-- The number of weeks available for the tournament --/
def available_weeks : ℕ := 4

/-- The number of matches each team plays in a double round-robin tournament --/
def matches_per_team (n : ℕ) : ℕ := 2 * (n - 1)

/-- The total number of matches in a double round-robin tournament --/
def total_matches (n : ℕ) : ℕ := n * (n - 1)

/-- The maximum number of away matches a team can play in the available weeks --/
def max_away_matches : ℕ := available_weeks

theorem double_round_robin_max_teams :
  ∀ n : ℕ, n ≤ max_teams ∧ 
  matches_per_team n ≤ 2 * max_away_matches ∧
  (∀ m : ℕ, m > max_teams → matches_per_team m > 2 * max_away_matches) :=
by sorry

#check double_round_robin_max_teams

end NUMINAMATH_CALUDE_double_round_robin_max_teams_l1439_143956


namespace NUMINAMATH_CALUDE_largest_rational_l1439_143904

theorem largest_rational (a b c d : ℚ) : 
  a = -1 → b = 0 → c = -3 → d = (8 : ℚ) / 100 → 
  max a (max b (max c d)) = d := by
  sorry

end NUMINAMATH_CALUDE_largest_rational_l1439_143904


namespace NUMINAMATH_CALUDE_no_quadruple_sum_2013_divisors_l1439_143953

theorem no_quadruple_sum_2013_divisors :
  ¬ (∃ (a b c d : ℕ+), 
      (a.val + b.val + c.val + d.val = 2013) ∧ 
      (2013 % a.val = 0) ∧ 
      (2013 % b.val = 0) ∧ 
      (2013 % c.val = 0) ∧ 
      (2013 % d.val = 0)) := by
  sorry

end NUMINAMATH_CALUDE_no_quadruple_sum_2013_divisors_l1439_143953


namespace NUMINAMATH_CALUDE_bob_wins_2033_alice_wins_2034_l1439_143917

/-- Represents the possible moves for each player -/
inductive Move
| Alice : (n : Nat) → n = 2 ∨ n = 5 → Move
| Bob : (n : Nat) → n = 1 ∨ n = 3 ∨ n = 4 → Move

/-- Represents the game state -/
structure GameState where
  coins : Nat
  aliceTurn : Bool

/-- Determines if the current state is a winning position for the player whose turn it is -/
def isWinningPosition (state : GameState) : Prop := sorry

/-- The game ends when there are no valid moves or Alice wins instantly -/
def gameOver (state : GameState) : Prop :=
  (state.coins < 1) ∨ (state.aliceTurn ∧ state.coins = 5)

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState := sorry

/-- Theorem stating that Bob has a winning strategy when starting with 2033 coins -/
theorem bob_wins_2033 :
  ∃ (strategy : GameState → Move),
    ∀ (aliceFirst : Bool),
      isWinningPosition (GameState.mk 2033 (¬aliceFirst)) := sorry

/-- Theorem stating that Alice has a winning strategy when starting with 2034 coins -/
theorem alice_wins_2034 :
  ∃ (strategy : GameState → Move),
    ∀ (aliceFirst : Bool),
      isWinningPosition (GameState.mk 2034 aliceFirst) := sorry

end NUMINAMATH_CALUDE_bob_wins_2033_alice_wins_2034_l1439_143917


namespace NUMINAMATH_CALUDE_final_top_number_is_16_l1439_143943

/-- Represents the state of the paper after folding operations -/
structure PaperState :=
  (top_number : Nat)

/-- Represents a folding operation -/
inductive FoldOperation
  | FoldBottomUp
  | FoldTopDown
  | FoldLeftRight

/-- The initial configuration of the paper -/
def initial_paper : List (List Nat) :=
  [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

/-- Perform a single fold operation -/
def fold (state : PaperState) (op : FoldOperation) : PaperState :=
  match op with
  | FoldOperation.FoldBottomUp => { top_number := 15 }
  | FoldOperation.FoldTopDown => { top_number := 9 }
  | FoldOperation.FoldLeftRight => { top_number := state.top_number + 1 }

/-- Perform a sequence of fold operations -/
def fold_sequence (initial : PaperState) (ops : List FoldOperation) : PaperState :=
  ops.foldl fold initial

/-- The theorem to be proved -/
theorem final_top_number_is_16 :
  (fold_sequence { top_number := 1 }
    [FoldOperation.FoldBottomUp,
     FoldOperation.FoldTopDown,
     FoldOperation.FoldBottomUp,
     FoldOperation.FoldLeftRight]).top_number = 16 := by
  sorry


end NUMINAMATH_CALUDE_final_top_number_is_16_l1439_143943


namespace NUMINAMATH_CALUDE_average_after_12th_innings_l1439_143916

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : ℕ
  lastScore : ℕ
  averageIncrease : ℕ
  neverNotOut : Bool

/-- Calculates the average score after the last innings -/
def averageAfterLastInnings (performance : BatsmanPerformance) : ℕ :=
  sorry

/-- Theorem stating the average after the 12th innings -/
theorem average_after_12th_innings (performance : BatsmanPerformance) 
  (h1 : performance.innings = 12)
  (h2 : performance.lastScore = 75)
  (h3 : performance.averageIncrease = 1)
  (h4 : performance.neverNotOut = true) :
  averageAfterLastInnings performance = 64 :=
sorry

end NUMINAMATH_CALUDE_average_after_12th_innings_l1439_143916


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_exists_l1439_143963

theorem polynomial_coefficient_sum_exists : ∃ (a b c d : ℤ),
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 - 3*x^2 + 12*x - 8) ∧
  (∃ (s : ℤ), s = a + b + c + d) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_exists_l1439_143963


namespace NUMINAMATH_CALUDE_min_sum_of_ten_numbers_l1439_143929

theorem min_sum_of_ten_numbers (S : Finset ℕ) : 
  S.card = 10 → 
  (∀ T ⊆ S, T.card = 5 → (T.prod id) % 2 = 0) → 
  (S.sum id) % 2 = 1 → 
  ∃ min_sum : ℕ, 
    (S.sum id = min_sum) ∧ 
    (∀ S' : Finset ℕ, S'.card = 10 → 
      (∀ T' ⊆ S', T'.card = 5 → (T'.prod id) % 2 = 0) → 
      (S'.sum id) % 2 = 1 → 
      S'.sum id ≥ min_sum) ∧
    min_sum = 51 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_ten_numbers_l1439_143929


namespace NUMINAMATH_CALUDE_snakes_not_hiding_l1439_143981

/-- Given a total of 95 snakes and 64 hiding snakes, prove that the number of snakes not hiding is 31. -/
theorem snakes_not_hiding (total_snakes : ℕ) (hiding_snakes : ℕ) 
  (h1 : total_snakes = 95) (h2 : hiding_snakes = 64) : 
  total_snakes - hiding_snakes = 31 := by
  sorry

end NUMINAMATH_CALUDE_snakes_not_hiding_l1439_143981


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1439_143978

theorem negation_of_universal_statement (S : Set ℝ) :
  (¬ ∀ x ∈ S, |x| > 1) ↔ (∃ x ∈ S, |x| ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1439_143978


namespace NUMINAMATH_CALUDE_simplify_expression_l1439_143997

theorem simplify_expression (w : ℝ) : -2*w + 3 - 4*w + 7 + 6*w - 5 - 8*w + 8 = -8*w + 13 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1439_143997


namespace NUMINAMATH_CALUDE_annie_thyme_pots_l1439_143990

/-- The number of pots of thyme Annie planted -/
def thyme_pots : ℕ := sorry

/-- The total number of leaves -/
def total_leaves : ℕ := 354

/-- The number of pots of basil -/
def basil_pots : ℕ := 3

/-- The number of pots of rosemary -/
def rosemary_pots : ℕ := 9

/-- The number of leaves per basil plant -/
def leaves_per_basil : ℕ := 4

/-- The number of leaves per rosemary plant -/
def leaves_per_rosemary : ℕ := 18

/-- The number of leaves per thyme plant -/
def leaves_per_thyme : ℕ := 30

theorem annie_thyme_pots : 
  thyme_pots = 6 :=
by sorry

end NUMINAMATH_CALUDE_annie_thyme_pots_l1439_143990


namespace NUMINAMATH_CALUDE_pastor_prayer_theorem_l1439_143903

/-- Represents the number of times Pastor Paul prays per day (except on Sundays) -/
def paul_prayers : ℕ := sorry

/-- Represents the number of times Pastor Bruce prays per day (except on Sundays) -/
def bruce_prayers : ℕ := sorry

/-- The total number of times Pastor Paul prays in a week -/
def paul_weekly_prayers : ℕ := 6 * paul_prayers + 2 * paul_prayers

/-- The total number of times Pastor Bruce prays in a week -/
def bruce_weekly_prayers : ℕ := 6 * (paul_prayers / 2) + 4 * paul_prayers

theorem pastor_prayer_theorem :
  paul_prayers = 20 ∧
  bruce_prayers = paul_prayers / 2 ∧
  paul_weekly_prayers = bruce_weekly_prayers + 20 := by
sorry

end NUMINAMATH_CALUDE_pastor_prayer_theorem_l1439_143903


namespace NUMINAMATH_CALUDE_inequality_proof_l1439_143940

theorem inequality_proof (a b c d : ℝ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1439_143940


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1439_143936

theorem quadratic_equation_solution (x : ℝ) :
  x^2 + 4*x - 2 = 0 ↔ x = -2 + Real.sqrt 6 ∨ x = -2 - Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1439_143936


namespace NUMINAMATH_CALUDE_two_numbers_problem_l1439_143919

theorem two_numbers_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 ∧ y = 17 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l1439_143919


namespace NUMINAMATH_CALUDE_sin_two_alpha_value_l1439_143979

theorem sin_two_alpha_value (α : Real) (h : Real.sin α + Real.cos α = 2/3) : 
  Real.sin (2 * α) = -5/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_value_l1439_143979


namespace NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_l1439_143941

theorem consecutive_integers_product_812_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_l1439_143941


namespace NUMINAMATH_CALUDE_key_chain_manufacturing_cost_l1439_143937

theorem key_chain_manufacturing_cost 
  (P : ℝ) -- Selling price
  (old_profit_percentage : ℝ) -- Old profit percentage
  (new_profit_percentage : ℝ) -- New profit percentage
  (new_manufacturing_cost : ℝ) -- New manufacturing cost
  (h1 : old_profit_percentage = 0.4) -- Old profit was 40%
  (h2 : new_profit_percentage = 0.5) -- New profit is 50%
  (h3 : new_manufacturing_cost = 50) -- New manufacturing cost is $50
  (h4 : P = new_manufacturing_cost / (1 - new_profit_percentage)) -- Selling price calculation
  : (1 - old_profit_percentage) * P = 60 := by
  sorry


end NUMINAMATH_CALUDE_key_chain_manufacturing_cost_l1439_143937


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1439_143915

theorem algebraic_expression_value (a b : ℝ) (h : a - b + 3 = 0) :
  2 - 3*a + 3*b = 11 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1439_143915


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l1439_143901

/-- The x-intercept of the line 4x + 6y = 24 is (6, 0) -/
theorem x_intercept_of_line (x y : ℝ) : 
  4 * x + 6 * y = 24 → y = 0 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l1439_143901


namespace NUMINAMATH_CALUDE_thabo_hardcover_nonfiction_count_l1439_143911

/-- Represents the number of books Thabo owns of each type -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def is_valid_collection (books : BookCollection) : Prop :=
  books.hardcover_nonfiction + books.paperback_nonfiction + books.paperback_fiction = 220 ∧
  books.paperback_nonfiction = books.hardcover_nonfiction + 20 ∧
  books.paperback_fiction = 2 * books.paperback_nonfiction

theorem thabo_hardcover_nonfiction_count :
  ∀ (books : BookCollection), is_valid_collection books → books.hardcover_nonfiction = 40 := by
  sorry

end NUMINAMATH_CALUDE_thabo_hardcover_nonfiction_count_l1439_143911


namespace NUMINAMATH_CALUDE_puppy_sale_cost_l1439_143955

theorem puppy_sale_cost (total_cost : ℕ) (non_sale_cost : ℕ) (num_puppies : ℕ) (num_non_sale : ℕ) :
  total_cost = 800 →
  non_sale_cost = 175 →
  num_puppies = 5 →
  num_non_sale = 2 →
  (total_cost - num_non_sale * non_sale_cost) / (num_puppies - num_non_sale) = 150 := by
  sorry

end NUMINAMATH_CALUDE_puppy_sale_cost_l1439_143955


namespace NUMINAMATH_CALUDE_animal_rescue_proof_l1439_143970

theorem animal_rescue_proof (sheep cows dogs pigs chickens rabbits ducks : ℕ) : 
  sheep = 20 ∧ cows = 10 ∧ dogs = 14 ∧ pigs = 8 ∧ chickens = 12 ∧ rabbits = 6 ∧ ducks = 15 →
  ∃ (saved_sheep saved_cows saved_dogs saved_pigs saved_chickens saved_rabbits saved_ducks : ℕ),
    saved_sheep = 14 ∧
    saved_cows = 6 ∧
    saved_dogs = 11 ∧
    saved_pigs = 6 ∧
    saved_chickens = 10 ∧
    saved_rabbits = 5 ∧
    saved_ducks = 10 ∧
    saved_sheep = sheep - (sheep * 3 / 10) ∧
    (cows - saved_cows) * (cows - saved_cows) = pigs - saved_pigs ∧
    saved_dogs = dogs * 3 / 4 ∧
    chickens - saved_chickens = (cows - saved_cows) / 2 ∧
    rabbits - saved_rabbits = 1 ∧
    saved_ducks = saved_rabbits * 2 ∧
    saved_sheep + saved_cows + saved_dogs + saved_pigs + saved_chickens + saved_rabbits + saved_ducks ≥ 50 :=
by
  sorry

end NUMINAMATH_CALUDE_animal_rescue_proof_l1439_143970


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l1439_143921

theorem existence_of_special_integers : ∃ (a b : ℕ+), 
  (¬ (7 ∣ (a.val * b.val * (a.val + b.val)))) ∧ 
  ((7^7 : ℕ) ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l1439_143921


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1439_143998

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (a₁ : ℝ), ∀ n, a n = a₁ * r ^ (n - 1)

-- Define the property of three terms forming a geometric sequence
def form_geometric_sequence (x y z : ℝ) : Prop :=
  y * y = x * z

-- Theorem statement
theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  form_geometric_sequence (a 3) (a 6) (a 9) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1439_143998


namespace NUMINAMATH_CALUDE_max_sum_abs_on_circle_l1439_143923

theorem max_sum_abs_on_circle :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧
  (∀ x y : ℝ, x^2 + y^2 = 4 → |x| + |y| ≤ M) ∧
  (∃ x y : ℝ, x^2 + y^2 = 4 ∧ |x| + |y| = M) := by
sorry

end NUMINAMATH_CALUDE_max_sum_abs_on_circle_l1439_143923


namespace NUMINAMATH_CALUDE_coefficient_value_l1439_143926

def P (c : ℝ) (x : ℝ) : ℝ := x^4 + 3*x^3 + 2*x^2 + c*x + 15

theorem coefficient_value (c : ℝ) :
  (∀ x, (x - 7 : ℝ) ∣ P c x) → c = -508 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_value_l1439_143926


namespace NUMINAMATH_CALUDE_stating_scale_theorem_l1439_143905

/-- Represents a curve in the xy-plane -/
structure Curve where
  equation : ℝ → ℝ → Prop

/-- Applies a scaling transformation to a curve in the y-axis direction -/
def scale_y (c : Curve) (k : ℝ) : Curve :=
  { equation := λ x y => c.equation x (y / k) }

/-- The original curve x^2 - 4y^2 = 16 -/
def original_curve : Curve :=
  { equation := λ x y => x^2 - 4*y^2 = 16 }

/-- The transformed curve x^2 - y^2 = 16 -/
def transformed_curve : Curve :=
  { equation := λ x y => x^2 - y^2 = 16 }

/-- 
Theorem stating that scaling the original curve by factor 2 in the y-direction 
results in the transformed curve
-/
theorem scale_theorem : scale_y original_curve 2 = transformed_curve := by
  sorry

end NUMINAMATH_CALUDE_stating_scale_theorem_l1439_143905


namespace NUMINAMATH_CALUDE_triangle_side_length_l1439_143980

theorem triangle_side_length (a b c : ℝ) (B : ℝ) : 
  a * c = 8 → a + c = 7 → B = π / 3 → b = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1439_143980


namespace NUMINAMATH_CALUDE_selection_with_both_genders_l1439_143965

/-- The number of ways to select 3 people from a group of 4 male students and 6 female students, 
    such that both male and female students are included. -/
theorem selection_with_both_genders (male_count : Nat) (female_count : Nat) : 
  male_count = 4 → female_count = 6 → 
  (Nat.choose (male_count + female_count) 3 - 
   Nat.choose male_count 3 - 
   Nat.choose female_count 3) = 96 := by
  sorry

end NUMINAMATH_CALUDE_selection_with_both_genders_l1439_143965


namespace NUMINAMATH_CALUDE_paul_failed_by_10_marks_l1439_143914

/-- Calculates the number of marks a student failed by in an exam -/
def marksFailed (maxMarks passingPercentage gotMarks : ℕ) : ℕ :=
  let passingMarks := (passingPercentage * maxMarks) / 100
  if gotMarks ≥ passingMarks then 0 else passingMarks - gotMarks

/-- Theorem stating that Paul failed by 10 marks -/
theorem paul_failed_by_10_marks :
  marksFailed 120 50 50 = 10 := by
  sorry

end NUMINAMATH_CALUDE_paul_failed_by_10_marks_l1439_143914


namespace NUMINAMATH_CALUDE_players_count_l1439_143913

/-- Represents the number of socks in each washing machine -/
structure SockCounts where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the number of players based on sock counts -/
def calculate_players (socks : SockCounts) : ℕ :=
  min socks.red (socks.blue + socks.green)

/-- Theorem stating that the number of players is 12 given the specific sock counts -/
theorem players_count (socks : SockCounts)
  (h_red : socks.red = 12)
  (h_blue : socks.blue = 10)
  (h_green : socks.green = 16) :
  calculate_players socks = 12 := by
  sorry

#eval calculate_players ⟨12, 10, 16⟩

end NUMINAMATH_CALUDE_players_count_l1439_143913


namespace NUMINAMATH_CALUDE_cube_root_simplification_and_rationalization_l1439_143991

theorem cube_root_simplification_and_rationalization :
  let x := (Real.rpow 6 (1/3)) / (Real.rpow 7 (1/3))
  let y := (Real.rpow 8 (1/3)) / (Real.rpow 9 (1/3))
  let z := (Real.rpow 10 (1/3)) / (Real.rpow 11 (1/3))
  x * y * z = (Real.rpow 223948320 (1/3)) / 693 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_and_rationalization_l1439_143991


namespace NUMINAMATH_CALUDE_factorization_proof_l1439_143969

theorem factorization_proof (x y : ℝ) : -3 * x^3 * y + 27 * x * y = -3 * x * y * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1439_143969


namespace NUMINAMATH_CALUDE_problem_1994_national_l1439_143972

theorem problem_1994_national (x y a : ℝ) 
  (h1 : x ∈ Set.Icc (-π/4) (π/4))
  (h2 : y ∈ Set.Icc (-π/4) (π/4))
  (h3 : x^3 + Real.sin x - 2*a = 0)
  (h4 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1994_national_l1439_143972


namespace NUMINAMATH_CALUDE_tens_digit_of_7_power_2011_l1439_143952

theorem tens_digit_of_7_power_2011 : 7^2011 % 100 = 43 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_7_power_2011_l1439_143952


namespace NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l1439_143928

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem smallest_prime_12_less_than_square : 
  ∃ (n : ℕ) (k : ℕ), 
    n > 0 ∧ 
    is_prime n ∧ 
    n = k^2 - 12 ∧ 
    ∀ (m : ℕ) (j : ℕ), m > 0 → is_prime m → m = j^2 - 12 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l1439_143928


namespace NUMINAMATH_CALUDE_milk_remaining_l1439_143984

theorem milk_remaining (initial : ℚ) (given_away : ℚ) (remaining : ℚ) :
  initial = 5.5 ∧ given_away = 17/4 → remaining = initial - given_away → remaining = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_milk_remaining_l1439_143984


namespace NUMINAMATH_CALUDE_orange_picking_ratio_l1439_143989

/-- Proves that the ratio of oranges picked on Tuesday to Monday is 3:1 --/
theorem orange_picking_ratio :
  let monday_oranges : ℕ := 100
  let wednesday_oranges : ℕ := 70
  let total_oranges : ℕ := 470
  let tuesday_oranges : ℕ := total_oranges - monday_oranges - wednesday_oranges
  tuesday_oranges / monday_oranges = 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_picking_ratio_l1439_143989


namespace NUMINAMATH_CALUDE_ordering_of_exponential_and_logarithm_l1439_143975

/-- Given a = e^0.1 - 1, b = 0.1, and c = ln 1.1, prove that a > b > c -/
theorem ordering_of_exponential_and_logarithm :
  let a := Real.exp 0.1 - 1
  let b := 0.1
  let c := Real.log 1.1
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ordering_of_exponential_and_logarithm_l1439_143975


namespace NUMINAMATH_CALUDE_joyce_farmland_l1439_143925

/-- Calculates the area of land suitable for growing vegetables given the size of the previous property, 
    the factor by which the new property is larger, and the size of a pond on the new property. -/
def land_for_vegetables (prev_property : ℝ) (size_factor : ℝ) (pond_size : ℝ) : ℝ :=
  prev_property * size_factor - pond_size

/-- Theorem stating that given a previous property of 2 acres, a new property 10 times larger, 
    and a 1-acre pond, the land suitable for growing vegetables is 19 acres. -/
theorem joyce_farmland : land_for_vegetables 2 10 1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_joyce_farmland_l1439_143925


namespace NUMINAMATH_CALUDE_shirt_boxes_per_roll_l1439_143948

-- Define the variables
def xl_boxes_per_roll : ℕ := 3
def shirt_boxes_to_wrap : ℕ := 20
def xl_boxes_to_wrap : ℕ := 12
def cost_per_roll : ℚ := 4
def total_cost : ℚ := 32

-- Define the theorem
theorem shirt_boxes_per_roll :
  ∃ (s : ℕ), 
    s * ((total_cost / cost_per_roll) - (xl_boxes_to_wrap / xl_boxes_per_roll)) = shirt_boxes_to_wrap ∧ 
    s = 5 := by
  sorry

end NUMINAMATH_CALUDE_shirt_boxes_per_roll_l1439_143948


namespace NUMINAMATH_CALUDE_fraction_1991_1949_position_l1439_143906

/-- Represents a fraction in the table -/
structure Fraction where
  numerator : ℕ
  denominator : ℕ

/-- Represents a row in the table -/
def Row := List Fraction

/-- Generates a row of the table given its index -/
def generateRow (n : ℕ) : Row :=
  sorry

/-- Checks if a fraction appears in a given row -/
def appearsInRow (f : Fraction) (r : Row) : Prop :=
  sorry

/-- The row number where 1991/1949 appears -/
def targetRow : ℕ := 3939

/-- The position of 1991/1949 in its row -/
def targetPosition : ℕ := 1949

theorem fraction_1991_1949_position : 
  let f := Fraction.mk 1991 1949
  let r := generateRow targetRow
  appearsInRow f r ∧ 
  (∃ (l1 l2 : List Fraction), r = l1 ++ [f] ++ l2 ∧ l1.length = targetPosition - 1) :=
sorry

end NUMINAMATH_CALUDE_fraction_1991_1949_position_l1439_143906


namespace NUMINAMATH_CALUDE_simple_interest_duration_l1439_143932

/-- Simple interest calculation -/
theorem simple_interest_duration
  (principal : ℝ)
  (rate : ℝ)
  (interest : ℝ)
  (h1 : principal = 69000)
  (h2 : rate = 50 / 3 / 100)  -- 16 2/3% converted to decimal
  (h3 : interest = 8625) :
  interest = principal * rate * (3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_duration_l1439_143932


namespace NUMINAMATH_CALUDE_average_age_first_fifth_dog_l1439_143907

/-- The age of the nth fastest dog -/
def dog_age (n : ℕ) : ℕ :=
  match n with
  | 1 => 10
  | 2 => dog_age 1 - 2
  | 3 => dog_age 2 + 4
  | 4 => dog_age 3 / 2
  | 5 => dog_age 4 + 20
  | _ => 0

theorem average_age_first_fifth_dog :
  (dog_age 1 + dog_age 5) / 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_age_first_fifth_dog_l1439_143907


namespace NUMINAMATH_CALUDE_billy_cherries_l1439_143931

theorem billy_cherries (cherries_eaten cherries_left : ℕ) 
  (h1 : cherries_eaten = 72)
  (h2 : cherries_left = 2) : 
  cherries_eaten + cherries_left = 74 := by
  sorry

end NUMINAMATH_CALUDE_billy_cherries_l1439_143931


namespace NUMINAMATH_CALUDE_book_cost_l1439_143961

theorem book_cost (n₅ n₃ : ℕ) : 
  (n₅ + n₃ > 10) → 
  (n₅ + n₃ < 20) → 
  (5 * n₅ = 3 * n₃) → 
  (5 * n₅ = 30) := by
sorry

end NUMINAMATH_CALUDE_book_cost_l1439_143961


namespace NUMINAMATH_CALUDE_simple_random_for_small_population_l1439_143993

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Other

/-- Determines the appropriate sampling method based on population size -/
def appropriateSamplingMethod (populationSize : ℕ) (sampleSize : ℕ) : SamplingMethod :=
  if populationSize ≤ 10 ∧ sampleSize = 1 then
    SamplingMethod.SimpleRandom
  else
    SamplingMethod.Other

/-- Theorem: For a population of 10 items with 1 item randomly selected,
    the appropriate sampling method is simple random sampling -/
theorem simple_random_for_small_population :
  appropriateSamplingMethod 10 1 = SamplingMethod.SimpleRandom :=
by sorry

end NUMINAMATH_CALUDE_simple_random_for_small_population_l1439_143993


namespace NUMINAMATH_CALUDE_john_needs_more_money_l1439_143910

/-- Given that John needs $2.5 in total and has $0.75, prove that he needs $1.75 more. -/
theorem john_needs_more_money (total_needed : ℝ) (amount_has : ℝ) 
  (h1 : total_needed = 2.5)
  (h2 : amount_has = 0.75) :
  total_needed - amount_has = 1.75 := by
sorry

end NUMINAMATH_CALUDE_john_needs_more_money_l1439_143910


namespace NUMINAMATH_CALUDE_min_coins_is_four_l1439_143939

/-- The minimum number of coins Ana can have -/
def min_coins : ℕ :=
  let initial_coins := 22
  let operations := [6, 18, -12]
  sorry

/-- Theorem: The minimum number of coins Ana can have is 4 -/
theorem min_coins_is_four : min_coins = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_coins_is_four_l1439_143939


namespace NUMINAMATH_CALUDE_square_nonnegative_is_universal_l1439_143947

/-- The proposition "The square of any real number is non-negative" -/
def square_nonnegative_prop : Prop := ∀ x : ℝ, x^2 ≥ 0

/-- Definition of a universal proposition -/
def is_universal_prop (P : Prop) : Prop := ∃ (α : Type) (Q : α → Prop), P = ∀ x : α, Q x

/-- The square_nonnegative_prop is a universal proposition -/
theorem square_nonnegative_is_universal : is_universal_prop square_nonnegative_prop := by sorry

end NUMINAMATH_CALUDE_square_nonnegative_is_universal_l1439_143947


namespace NUMINAMATH_CALUDE_scatter_plot_correlation_distinction_l1439_143951

/-- Represents a scatter plot --/
structure ScatterPlot where
  data : Set (ℝ × ℝ)

/-- Represents the correlation type in a scatter plot --/
inductive Correlation
  | Positive
  | Negative
  | Indeterminate

/-- Function to determine the correlation type from a scatter plot --/
def determineCorrelation (plot : ScatterPlot) : Correlation :=
  sorry

/-- Theorem stating that a scatter plot allows distinguishing between positive and negative correlations --/
theorem scatter_plot_correlation_distinction (plot : ScatterPlot) :
  ∃ (c : Correlation), c ≠ Correlation.Indeterminate :=
by
  sorry

end NUMINAMATH_CALUDE_scatter_plot_correlation_distinction_l1439_143951


namespace NUMINAMATH_CALUDE_cat_ratio_l1439_143900

theorem cat_ratio (melanie_cats jacob_cats : ℕ) 
  (melanie_twice_annie : melanie_cats = 2 * (melanie_cats / 2))
  (jacob_has_90 : jacob_cats = 90)
  (melanie_has_60 : melanie_cats = 60) :
  (melanie_cats / 2) / jacob_cats = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cat_ratio_l1439_143900


namespace NUMINAMATH_CALUDE_solution_is_axes_l1439_143966

def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - p.2)^2 = p.1^2 + p.2^2}

def x_axis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

def y_axis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0}

theorem solution_is_axes : solution_set = x_axis ∪ y_axis := by
  sorry

end NUMINAMATH_CALUDE_solution_is_axes_l1439_143966


namespace NUMINAMATH_CALUDE_distance_between_points_l1439_143950

/-- The distance between points (2,3) and (5,9) is 3√5. -/
theorem distance_between_points : Real.sqrt ((5 - 2)^2 + (9 - 3)^2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1439_143950


namespace NUMINAMATH_CALUDE_circumcircumcircumcoronene_tilings_l1439_143909

/-- Represents a tiling of a hexagon with edge length n using diamonds of side 1 -/
def HexagonTiling (n : ℕ) : Type := Unit

/-- The number of valid tilings for a hexagon with edge length n -/
def count_tilings (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of tilings for a hexagon with edge length 5 is 267227532 -/
theorem circumcircumcircumcoronene_tilings :
  count_tilings 5 = 267227532 := by sorry

end NUMINAMATH_CALUDE_circumcircumcircumcoronene_tilings_l1439_143909


namespace NUMINAMATH_CALUDE_hospital_worker_count_l1439_143982

theorem hospital_worker_count 
  (total_workers : ℕ) 
  (chosen_workers : ℕ) 
  (specific_pair_prob : ℚ) : 
  total_workers = 8 → 
  chosen_workers = 2 → 
  specific_pair_prob = 1 / 28 → 
  total_workers - 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_hospital_worker_count_l1439_143982


namespace NUMINAMATH_CALUDE_A_subset_B_l1439_143977

def A : Set ℝ := {x | |x - 2| < 1}
def B : Set ℝ := {x | (x - 1) * (x - 4) < 0}

theorem A_subset_B : A ⊆ B := by sorry

end NUMINAMATH_CALUDE_A_subset_B_l1439_143977


namespace NUMINAMATH_CALUDE_two_phase_tournament_matches_l1439_143968

/-- Calculate the number of matches in a round-robin tournament -/
def roundRobinMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of matches in a two-phase tennis tournament -/
theorem two_phase_tournament_matches : 
  roundRobinMatches 10 + roundRobinMatches 5 = 55 := by
  sorry

end NUMINAMATH_CALUDE_two_phase_tournament_matches_l1439_143968


namespace NUMINAMATH_CALUDE_factorial_sum_division_l1439_143927

theorem factorial_sum_division (n : ℕ) : (Nat.factorial 8 + Nat.factorial 9) / Nat.factorial 6 = 560 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_division_l1439_143927


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l1439_143942

theorem angle_sum_around_point (x : ℝ) : 
  (6 * x + 3 * x + x + 5 * x = 360) → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l1439_143942


namespace NUMINAMATH_CALUDE_problem_solution_l1439_143960

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 6)
  (h_eq2 : y + 1 / x = 30) :
  z + 1 / y = 38 / 179 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1439_143960


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1439_143999

theorem interest_rate_calculation (initial_amount loan_amount final_amount : ℚ) :
  initial_amount = 30 →
  loan_amount = 15 →
  final_amount = 33 →
  (final_amount - initial_amount) / loan_amount * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1439_143999


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_root_less_than_two_iff_l1439_143985

/-- The quadratic equation x^2 - (k+4)x + 4k = 0 -/
def quadratic (k x : ℝ) : ℝ := x^2 - (k+4)*x + 4*k

theorem quadratic_two_real_roots (k : ℝ) : 
  ∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0 :=
sorry

theorem root_less_than_two_iff (k : ℝ) : 
  (∃ x : ℝ, quadratic k x = 0 ∧ x < 2) ↔ k < 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_root_less_than_two_iff_l1439_143985


namespace NUMINAMATH_CALUDE_cube_root_scaling_l1439_143983

theorem cube_root_scaling (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : a^(1/3) = 2.938) (h2 : b^(1/3) = 6.329) (h3 : c = 253600) 
  (h4 : b = 10 * a) (h5 : c = 1000 * b) : 
  c^(1/3) = 63.29 := by sorry

end NUMINAMATH_CALUDE_cube_root_scaling_l1439_143983


namespace NUMINAMATH_CALUDE_launch_vehicle_ratio_l1439_143938

/-- Represents a three-stage cylindrical launch vehicle -/
structure LaunchVehicle where
  l₁ : ℝ  -- Length of the first stage
  l₂ : ℝ  -- Length of the second (middle) stage
  l₃ : ℝ  -- Length of the third stage

/-- The conditions for the launch vehicle -/
def LaunchVehicleConditions (v : LaunchVehicle) : Prop :=
  v.l₁ > 0 ∧ v.l₂ > 0 ∧ v.l₃ > 0 ∧
  v.l₂ = (v.l₁ + v.l₃) / 2 ∧
  v.l₂^3 = (6 / 13) * (v.l₁^3 + v.l₃^3)

/-- The theorem stating the ratio of lengths of first and third stages -/
theorem launch_vehicle_ratio (v : LaunchVehicle) 
  (h : LaunchVehicleConditions v) : v.l₁ / v.l₃ = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_launch_vehicle_ratio_l1439_143938


namespace NUMINAMATH_CALUDE_systematic_sampling_characterization_l1439_143945

/-- Represents a population in a sampling context -/
structure Population where
  size : ℕ
  is_large : Prop

/-- Represents a sampling method -/
structure SamplingMethod where
  divides_population : Prop
  uses_predetermined_rule : Prop
  selects_one_per_part : Prop

/-- Definition of systematic sampling -/
def systematic_sampling (pop : Population) (method : SamplingMethod) : Prop :=
  pop.is_large ∧ 
  method.divides_population ∧ 
  method.uses_predetermined_rule ∧ 
  method.selects_one_per_part

/-- Theorem stating the characterization of systematic sampling -/
theorem systematic_sampling_characterization 
  (pop : Population) 
  (method : SamplingMethod) : 
  systematic_sampling pop method ↔ 
    (method.divides_population ∧ 
     method.uses_predetermined_rule ∧ 
     method.selects_one_per_part) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_characterization_l1439_143945


namespace NUMINAMATH_CALUDE_average_score_bounds_l1439_143988

def score_distribution : List (ℕ × ℕ) :=
  [(100, 2), (90, 9), (80, 17), (70, 28), (60, 36), (50, 7), (48, 1)]

def total_students : ℕ := (score_distribution.map Prod.snd).sum

def min_score_sum : ℕ := (score_distribution.map (λ (s, n) => s * n)).sum

def max_score_sum : ℕ := (score_distribution.map (λ (s, n) => 
  if s = 100 then s * n else (s + 9) * n)).sum

theorem average_score_bounds :
  (min_score_sum : ℚ) / total_students > 68 ∧
  (max_score_sum : ℚ) / total_students < 78 := by
  sorry

end NUMINAMATH_CALUDE_average_score_bounds_l1439_143988


namespace NUMINAMATH_CALUDE_nabla_problem_l1439_143922

-- Define the ∇ operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- Theorem statement
theorem nabla_problem : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end NUMINAMATH_CALUDE_nabla_problem_l1439_143922
