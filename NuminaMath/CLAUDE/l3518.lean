import Mathlib

namespace robin_gum_total_l3518_351867

theorem robin_gum_total (packages : ℕ) (pieces_per_package : ℕ) (extra_pieces : ℕ) : 
  packages = 5 → pieces_per_package = 7 → extra_pieces = 6 →
  packages * pieces_per_package + extra_pieces = 41 := by
  sorry

end robin_gum_total_l3518_351867


namespace notebook_distribution_l3518_351829

theorem notebook_distribution (x : ℕ) : 
  (x > 0) → 
  ((x - 1) % 3 = 0) → 
  ((x + 2) % 4 = 0) → 
  ((x - 1) / 3 : ℚ) = ((x + 2) / 4 : ℚ) :=
by sorry

end notebook_distribution_l3518_351829


namespace wolf_winning_strategy_wolf_wins_l3518_351839

/-- Represents a player in the game -/
inductive Player
| Wolf
| Hare

/-- Represents the state of the game board -/
structure GameState where
  number : Nat
  currentPlayer : Player

/-- Defines a valid move in the game -/
def isValidMove (n : Nat) (digit : Nat) : Prop :=
  digit > 0 ∧ digit ≤ 9 ∧ digit ≤ n

/-- Applies a move to the game state -/
def applyMove (state : GameState) (digit : Nat) : GameState :=
  { number := state.number - digit,
    currentPlayer := match state.currentPlayer with
      | Player.Wolf => Player.Hare
      | Player.Hare => Player.Wolf }

/-- Defines the winning condition -/
def isWinningState (state : GameState) : Prop :=
  state.number = 0

/-- Theorem: There exists a winning strategy for Wolf starting with 1234 -/
theorem wolf_winning_strategy :
  ∃ (strategy : GameState → Nat),
    (∀ (state : GameState), isValidMove state.number (strategy state)) →
    (∀ (state : GameState),
      state.currentPlayer = Player.Wolf →
      isWinningState (applyMove state (strategy state)) ∨
      ∃ (hareMove : Nat),
        isValidMove (applyMove state (strategy state)).number hareMove →
        isWinningState (applyMove (applyMove state (strategy state)) hareMove)) :=
sorry

/-- The initial game state -/
def initialState : GameState :=
  { number := 1234, currentPlayer := Player.Wolf }

/-- Corollary: Wolf wins the game starting from 1234 -/
theorem wolf_wins : ∃ (moves : List Nat), 
  isWinningState (moves.foldl applyMove initialState) ∧
  moves.length % 2 = 0 :=
sorry

end wolf_winning_strategy_wolf_wins_l3518_351839


namespace consecutive_sum_formula_l3518_351819

def consecutive_sum (n : ℤ) : ℤ := (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)

theorem consecutive_sum_formula (n : ℤ) : consecutive_sum n = 5 * n + 20 := by
  sorry

end consecutive_sum_formula_l3518_351819


namespace seedling_purchase_solution_l3518_351835

/-- Represents the unit prices and maximum purchase of seedlings --/
structure SeedlingPurchase where
  price_a : ℝ  -- Unit price of type A seedlings
  price_b : ℝ  -- Unit price of type B seedlings
  max_a : ℕ    -- Maximum number of type A seedlings that can be purchased

/-- Theorem statement for the seedling purchase problem --/
theorem seedling_purchase_solution :
  ∃ (sp : SeedlingPurchase),
    -- Condition 1: 30 bundles of A and 10 bundles of B cost 380 yuan
    30 * sp.price_a + 10 * sp.price_b = 380 ∧
    -- Condition 2: 50 bundles of A and 30 bundles of B cost 740 yuan
    50 * sp.price_a + 30 * sp.price_b = 740 ∧
    -- Condition 3: Budget constraint with discount
    sp.price_a * 0.9 * sp.max_a + sp.price_b * 0.9 * (100 - sp.max_a) ≤ 828 ∧
    -- Solution 1: Unit prices
    sp.price_a = 10 ∧ sp.price_b = 8 ∧
    -- Solution 2: Maximum number of type A seedlings
    sp.max_a = 60 := by
  sorry

end seedling_purchase_solution_l3518_351835


namespace similar_triangles_shortest_side_l3518_351852

/-- Given two similar right triangles, where one has sides 7, 24, and 25 inches,
    and the other has a hypotenuse of 100 inches, the shortest side of the larger triangle is 28 inches. -/
theorem similar_triangles_shortest_side (a b c d e : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
  a^2 + b^2 = c^2 →
  a = 7 →
  b = 24 →
  c = 25 →
  e = 100 →
  d / a = e / c →
  d = 28 := by
sorry

end similar_triangles_shortest_side_l3518_351852


namespace cos_alpha_value_l3518_351882

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (π/6 + α) = 3/5)
  (h2 : π/3 < α ∧ α < 5*π/6) : 
  Real.cos α = (3 - 4 * Real.sqrt 3) / 10 := by sorry

end cos_alpha_value_l3518_351882


namespace value_standard_deviations_below_mean_l3518_351823

/-- For a normal distribution with mean 14.5 and standard deviation 1.5,
    the value 11.5 is 2 standard deviations less than the mean. -/
theorem value_standard_deviations_below_mean
  (μ : ℝ) (σ : ℝ) (x : ℝ)
  (h_mean : μ = 14.5)
  (h_std_dev : σ = 1.5)
  (h_value : x = 11.5) :
  (μ - x) / σ = 2 := by
sorry

end value_standard_deviations_below_mean_l3518_351823


namespace outfit_choices_l3518_351803

/-- The number of colors available for each clothing item -/
def num_colors : ℕ := 8

/-- The number of shirts available -/
def num_shirts : ℕ := num_colors

/-- The number of pants available -/
def num_pants : ℕ := num_colors

/-- The number of hats available -/
def num_hats : ℕ := num_colors

/-- Calculate the total number of outfit combinations -/
def total_combinations : ℕ := num_shirts * num_pants * num_hats

/-- Calculate the number of outfits where shirt and pants are the same color -/
def same_color_combinations : ℕ := num_colors * num_hats

/-- Calculate the number of valid outfit choices -/
def valid_outfits : ℕ := total_combinations - same_color_combinations

theorem outfit_choices : valid_outfits = 448 := by
  sorry

end outfit_choices_l3518_351803


namespace parabola_chord_intersection_l3518_351878

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16*x

-- Define a point on the parabola
def point_on_parabola (p : ℝ × ℝ) : Prop := parabola p.1 p.2

-- Define perpendicular vectors
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Theorem statement
theorem parabola_chord_intersection :
  ∀ (A B : ℝ × ℝ),
  point_on_parabola A →
  point_on_parabola B →
  perpendicular A B →
  ∃ (t : ℝ), A.1 = t * A.2 + 16 ∧ B.1 = t * B.2 + 16 :=
sorry

end parabola_chord_intersection_l3518_351878


namespace sqrt_450_simplified_l3518_351826

theorem sqrt_450_simplified : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplified_l3518_351826


namespace equation_holds_l3518_351880

theorem equation_holds (a b c : ℤ) (h1 : a = c + 1) (h2 : b = a - 1) :
  a * (a - b) + b * (b - c) + c * (c - a) = 2 := by
  sorry

end equation_holds_l3518_351880


namespace square_area_problem_l3518_351841

theorem square_area_problem (a b : ℝ) (h : a > b) :
  let diagonal_I := a - b
  let area_I := (diagonal_I^2) / 2
  let area_II := 3 * area_I
  area_II = (3 * (a - b)^2) / 2 := by
sorry

end square_area_problem_l3518_351841


namespace initial_savings_theorem_l3518_351846

def calculate_initial_savings (repair_fee : ℕ) (remaining_savings : ℕ) : ℕ :=
  let corner_light := 2 * repair_fee
  let brake_disk := 3 * corner_light
  let floor_mats := brake_disk
  let steering_wheel_cover := corner_light / 2
  let seat_covers := 2 * floor_mats
  let total_expenses := repair_fee + corner_light + 2 * brake_disk + floor_mats + steering_wheel_cover + seat_covers
  remaining_savings + total_expenses

theorem initial_savings_theorem (repair_fee : ℕ) (remaining_savings : ℕ) :
  repair_fee = 10 ∧ remaining_savings = 480 →
  calculate_initial_savings repair_fee remaining_savings = 820 :=
by sorry

end initial_savings_theorem_l3518_351846


namespace bryson_new_shoes_l3518_351877

/-- Proves that buying 2 pairs of shoes results in 4 new shoes -/
theorem bryson_new_shoes : 
  ∀ (pairs_bought : ℕ) (shoes_per_pair : ℕ),
  pairs_bought = 2 → shoes_per_pair = 2 →
  pairs_bought * shoes_per_pair = 4 := by
  sorry

end bryson_new_shoes_l3518_351877


namespace algebraic_simplification_l3518_351827

theorem algebraic_simplification (m n : ℝ) :
  (3 * m^2 - m * n + 5) - 2 * (5 * m * n - 4 * m^2 + 2) = 11 * m^2 - 11 * m * n + 1 := by
  sorry

end algebraic_simplification_l3518_351827


namespace polyhedron_surface_area_l3518_351821

/-- Represents a polyhedron with three orthographic views --/
structure Polyhedron where
  front_view : Set (ℝ × ℝ)
  side_view : Set (ℝ × ℝ)
  top_view : Set (ℝ × ℝ)

/-- Calculates the surface area of a polyhedron --/
noncomputable def surface_area (p : Polyhedron) : ℝ := sorry

/-- Theorem stating that the surface area of the given polyhedron is 8 --/
theorem polyhedron_surface_area (p : Polyhedron) : surface_area p = 8 := by sorry

end polyhedron_surface_area_l3518_351821


namespace quadratic_equivalence_l3518_351888

theorem quadratic_equivalence : 
  ∀ x y : ℝ, y = x^2 - 2*x + 3 ↔ y = (x - 1)^2 + 2 := by sorry

end quadratic_equivalence_l3518_351888


namespace mario_orange_consumption_l3518_351837

/-- Represents the amount of fruit eaten by each person in ounces -/
structure FruitConsumption where
  mario : ℕ
  lydia : ℕ
  nicolai : ℕ

/-- Converts pounds to ounces -/
def poundsToOunces (pounds : ℕ) : ℕ := pounds * 16

/-- Theorem: Given the conditions, Mario ate 8 ounces of oranges -/
theorem mario_orange_consumption (total : ℕ) (fc : FruitConsumption) 
  (h1 : poundsToOunces total = fc.mario + fc.lydia + fc.nicolai)
  (h2 : total = 8)
  (h3 : fc.lydia = 24)
  (h4 : fc.nicolai = poundsToOunces 6) :
  fc.mario = 8 := by
  sorry

#check mario_orange_consumption

end mario_orange_consumption_l3518_351837


namespace rectangular_prism_parallel_edges_l3518_351822

/-- A rectangular prism with integer dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of pairs of parallel edges in a rectangular prism -/
def parallel_edge_pairs (prism : RectangularPrism) : ℕ :=
  4 * 3

theorem rectangular_prism_parallel_edges :
  ∀ (prism : RectangularPrism),
    prism.length = 4 ∧ prism.width = 3 ∧ prism.height = 2 →
    parallel_edge_pairs prism = 12 := by
  sorry

#check rectangular_prism_parallel_edges

end rectangular_prism_parallel_edges_l3518_351822


namespace total_squares_6x6_grid_l3518_351872

/-- The number of squares of a given size in a grid --/
def count_squares (grid_size : ℕ) (square_size : ℕ) : ℕ :=
  (grid_size - square_size + 1) ^ 2

/-- The total number of squares in a 6x6 grid --/
theorem total_squares_6x6_grid :
  let grid_size := 6
  let square_sizes := [1, 2, 3, 4]
  (square_sizes.map (count_squares grid_size)).sum = 54 := by
  sorry

end total_squares_6x6_grid_l3518_351872


namespace house_tower_difference_l3518_351836

/-- Represents the number of blocks Randy used for different purposes -/
structure BlockCounts where
  total : ℕ
  house : ℕ
  tower : ℕ

/-- Theorem stating the difference in blocks used for house and tower -/
theorem house_tower_difference (randy : BlockCounts)
  (h1 : randy.total = 90)
  (h2 : randy.house = 89)
  (h3 : randy.tower = 63) :
  randy.house - randy.tower = 26 := by
  sorry

end house_tower_difference_l3518_351836


namespace intersection_of_A_and_B_l3518_351899

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def B : Set ℕ := {x | 2 ≤ x ∧ x < 6}

theorem intersection_of_A_and_B : A ∩ B = {2, 3, 4, 5} := by sorry

end intersection_of_A_and_B_l3518_351899


namespace max_a_value_l3518_351866

theorem max_a_value (x a : ℤ) : 
  (∃ x : ℤ, x^2 + a*x = -30) → 
  (a > 0) → 
  a ≤ 31 :=
by sorry

end max_a_value_l3518_351866


namespace x₁_plus_x₂_pos_l3518_351843

noncomputable section

variables (a : ℝ) (x x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := Real.log (a * x + 1) - a * x - Real.log a

axiom a_pos : a > 0

axiom x_domain : x > -1/a

axiom x₁_domain : -1/a < x₁ ∧ x₁ < 0

axiom x₂_domain : x₂ > 0

axiom f_roots : f a x₁ = 0 ∧ f a x₂ = 0

theorem x₁_plus_x₂_pos : x₁ + x₂ > 0 := by sorry

end x₁_plus_x₂_pos_l3518_351843


namespace extreme_value_implies_a_l3518_351833

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem extreme_value_implies_a (a : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), |x + 3| < ε → f a x ≤ f a (-3)) →
  a = 5 := by
  sorry

end extreme_value_implies_a_l3518_351833


namespace cos_shift_equivalence_l3518_351859

open Real

theorem cos_shift_equivalence (x : ℝ) :
  cos (2 * (x + π / 6) - π / 3) = cos (2 * x) :=
by sorry

end cos_shift_equivalence_l3518_351859


namespace probability_one_heads_three_coins_l3518_351808

theorem probability_one_heads_three_coins :
  let n : ℕ := 3  -- number of coins
  let p : ℚ := 1/2  -- probability of heads for a fair coin
  let k : ℕ := 1  -- number of heads we want
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k) = 3/8 :=
by sorry

end probability_one_heads_three_coins_l3518_351808


namespace polynomial_form_l3518_351854

/-- A polynomial that satisfies the given condition -/
noncomputable def satisfying_polynomial (P : ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), P (a + b - 2*c) + P (b + c - 2*a) + P (c + a - 2*b) = 
                  3*P (a - b) + 3*P (b - c) + 3*P (c - a)

/-- The theorem stating the form of polynomials satisfying the condition -/
theorem polynomial_form (P : ℝ → ℝ) (hP : satisfying_polynomial P) :
  ∃ (a b : ℝ), ∀ x, P x = a * x^2 + b * x :=
sorry

end polynomial_form_l3518_351854


namespace mn_sum_for_5000_l3518_351804

theorem mn_sum_for_5000 (m n : ℕ+) : 
  m * n = 5000 →
  ¬(10 ∣ m) →
  ¬(10 ∣ n) →
  m + n = 633 := by
sorry

end mn_sum_for_5000_l3518_351804


namespace courtyard_length_is_20_l3518_351813

/-- The width of the courtyard in meters -/
def courtyard_width : ℝ := 16.5

/-- The number of paving stones required to cover the courtyard -/
def num_paving_stones : ℕ := 66

/-- The length of a paving stone in meters -/
def paving_stone_length : ℝ := 2.5

/-- The width of a paving stone in meters -/
def paving_stone_width : ℝ := 2

/-- The theorem stating that the length of the courtyard is 20 meters -/
theorem courtyard_length_is_20 : 
  (courtyard_width * (num_paving_stones * paving_stone_length * paving_stone_width) / courtyard_width) = 20 := by
  sorry

end courtyard_length_is_20_l3518_351813


namespace g_of_6_l3518_351825

def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 37*x^2 - 18*x - 80

theorem g_of_6 : g 6 = 712 := by
  sorry

end g_of_6_l3518_351825


namespace soccer_ball_inflation_time_l3518_351840

/-- The time in minutes it takes to inflate one soccer ball -/
def inflationTime : ℕ := 20

/-- The number of balls Alexia inflates -/
def alexiaBalls : ℕ := 20

/-- The number of additional balls Ermias inflates compared to Alexia -/
def ermiasDifference : ℕ := 5

/-- The total number of balls Ermias inflates -/
def ermiasBalls : ℕ := alexiaBalls + ermiasDifference

/-- The total number of balls inflated by both Alexia and Ermias -/
def totalBalls : ℕ := alexiaBalls + ermiasBalls

/-- The total time in minutes taken to inflate all soccer balls -/
def totalTime : ℕ := totalBalls * inflationTime

theorem soccer_ball_inflation_time : totalTime = 900 := by
  sorry

end soccer_ball_inflation_time_l3518_351840


namespace perfect_square_condition_l3518_351844

/-- If x^2 + mx + n is a perfect square, then n = (|m| / 2)^2 -/
theorem perfect_square_condition (m n : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 + m*x + n = k^2) → n = (|m| / 2)^2 := by
sorry

end perfect_square_condition_l3518_351844


namespace maxwell_brad_meeting_time_l3518_351816

/-- Proves that Maxwell walks for 8 hours before meeting Brad given the specified conditions -/
theorem maxwell_brad_meeting_time :
  let distance_between_homes : ℝ := 74
  let maxwell_speed : ℝ := 4
  let brad_speed : ℝ := 6
  let brad_delay : ℝ := 1

  let meeting_time : ℝ := 
    (distance_between_homes - maxwell_speed * brad_delay) / (maxwell_speed + brad_speed)

  let maxwell_total_time : ℝ := meeting_time + brad_delay

  maxwell_total_time = 8 := by
  sorry

end maxwell_brad_meeting_time_l3518_351816


namespace cube_root_of_2197_l3518_351897

theorem cube_root_of_2197 (x : ℝ) (h1 : x > 0) (h2 : x^3 = 2197) : x = 13 := by
  sorry

end cube_root_of_2197_l3518_351897


namespace simplify_absolute_expression_l3518_351869

theorem simplify_absolute_expression : abs (-4^2 - 3 + 6) = 13 := by
  sorry

end simplify_absolute_expression_l3518_351869


namespace union_A_B_when_m_1_intersection_A_B_empty_union_A_B_equals_A_l3518_351861

-- Define sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 2}
def B (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ 2 * m + 1}

-- Theorem 1: Union of A and B when m = 1
theorem union_A_B_when_m_1 :
  A ∪ B 1 = {x | -2 < x ∧ x ≤ 3} := by sorry

-- Theorem 2: Condition for A ∩ B = ∅
theorem intersection_A_B_empty (m : ℝ) :
  A ∩ B m = ∅ ↔ m ≤ -3/2 ∨ m ≥ 4 := by sorry

-- Theorem 3: Condition for A ∪ B = A
theorem union_A_B_equals_A (m : ℝ) :
  A ∪ B m = A ↔ m < -3 ∨ (0 < m ∧ m < 1/2) := by sorry

end union_A_B_when_m_1_intersection_A_B_empty_union_A_B_equals_A_l3518_351861


namespace vector_equation_solution_l3518_351892

theorem vector_equation_solution (α β : Real) :
  let A : Real × Real := (Real.cos α, Real.sin α)
  let B : Real × Real := (Real.cos β, Real.sin β)
  let C : Real × Real := (1/2, Real.sqrt 3/2)
  (C.1 = B.1 - A.1 ∧ C.2 = B.2 - A.2) → β = 2*Real.pi/3 ∨ β = 0 := by
  sorry

end vector_equation_solution_l3518_351892


namespace inequality_of_distinct_positive_numbers_l3518_351802

theorem inequality_of_distinct_positive_numbers (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hcd : c ≠ d) (hda : d ≠ a) :
  a^2 / b + b^2 / c + c^2 / d + d^2 / a > a + b + c + d :=
sorry

end inequality_of_distinct_positive_numbers_l3518_351802


namespace arithmetic_sequence_problem_l3518_351896

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a) 
    (h_sum : a 3 + a 8 = 10) : 
  3 * a 5 + a 7 = 20 := by
sorry

end arithmetic_sequence_problem_l3518_351896


namespace athletes_meeting_distance_l3518_351871

theorem athletes_meeting_distance (v₁ v₂ : ℝ) (h₁ : v₁ > 0) (h₂ : v₂ > 0) : 
  (∃ x : ℝ, x > 0 ∧ 
    300 / v₁ = (x - 300) / v₂ ∧ 
    (x + 100) / v₁ = (x - 100) / v₂) → 
  (∃ x : ℝ, x = 500) :=
by sorry

end athletes_meeting_distance_l3518_351871


namespace perfect_line_statistics_l3518_351894

/-- A scatter plot represents a set of points in a 2D plane. -/
structure ScatterPlot where
  points : Set (ℝ × ℝ)

/-- A straight line in 2D space. -/
structure StraightLine where
  slope : ℝ
  intercept : ℝ

/-- The sum of squared residuals for a scatter plot and a fitted line. -/
def sumSquaredResiduals (plot : ScatterPlot) (line : StraightLine) : ℝ := sorry

/-- The correlation coefficient for a scatter plot. -/
def correlationCoefficient (plot : ScatterPlot) : ℝ := sorry

/-- Predicate to check if all points in a scatter plot lie on a given straight line. -/
def allPointsOnLine (plot : ScatterPlot) (line : StraightLine) : Prop := sorry

theorem perfect_line_statistics (plot : ScatterPlot) (line : StraightLine) :
  allPointsOnLine plot line →
  sumSquaredResiduals plot line = 0 ∧ correlationCoefficient plot = 1 := by
  sorry

end perfect_line_statistics_l3518_351894


namespace reporter_earnings_per_hour_l3518_351887

/-- Calculate reporter's earnings per hour given their pay rate and work conditions --/
theorem reporter_earnings_per_hour 
  (words_per_minute : ℕ)
  (pay_per_word : ℚ)
  (pay_per_article : ℕ)
  (num_articles : ℕ)
  (total_hours : ℕ)
  (h1 : words_per_minute = 10)
  (h2 : pay_per_word = 1/10)
  (h3 : pay_per_article = 60)
  (h4 : num_articles = 3)
  (h5 : total_hours = 4) :
  (words_per_minute * 60 * total_hours : ℚ) * pay_per_word + 
  (num_articles * pay_per_article : ℚ) / total_hours = 105 := by
  sorry

#eval (10 * 60 * 4 : ℚ) * (1/10) + (3 * 60 : ℚ) / 4

end reporter_earnings_per_hour_l3518_351887


namespace find_divisor_find_divisor_proof_l3518_351805

theorem find_divisor (original : Nat) (subtracted : Nat) (divisor : Nat) : Prop :=
  let remaining := original - subtracted
  (original = 1387) →
  (subtracted = 7) →
  (remaining % divisor = 0) →
  (∀ d : Nat, d > divisor → remaining % d ≠ 0 ∨ (original - d) % d ≠ 0) →
  divisor = 23

-- The proof would go here
theorem find_divisor_proof : find_divisor 1387 7 23 := by
  sorry

end find_divisor_find_divisor_proof_l3518_351805


namespace quadratic_equation_coefficient_l3518_351815

theorem quadratic_equation_coefficient : ∀ a b c : ℝ,
  (∀ x, 3 * x^2 + 1 = 6 * x ↔ a * x^2 + b * x + c = 0) →
  a = 3 →
  b = -6 := by
  sorry

end quadratic_equation_coefficient_l3518_351815


namespace infinitely_many_primes_l3518_351864

theorem infinitely_many_primes : ∀ S : Finset Nat, (∀ p ∈ S, Nat.Prime p) → ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end infinitely_many_primes_l3518_351864


namespace no_valid_cd_l3518_351886

theorem no_valid_cd : ¬ ∃ (C D : ℕ+), 
  (Nat.lcm C D = 210) ∧ 
  (C : ℚ) / (D : ℚ) = 4 / 7 := by
sorry

end no_valid_cd_l3518_351886


namespace chemistry_class_size_l3518_351806

theorem chemistry_class_size 
  (total_students : ℕ) 
  (both_subjects : ℕ) 
  (h1 : total_students = 52) 
  (h2 : both_subjects = 8) 
  (h3 : ∃ (biology_only chemistry_only : ℕ), 
    total_students = biology_only + chemistry_only + both_subjects ∧
    chemistry_only + both_subjects = 2 * (biology_only + both_subjects)) :
  ∃ (chemistry_class : ℕ), chemistry_class = 40 ∧ 
    chemistry_class = (total_students - both_subjects) / 3 * 2 + both_subjects :=
by
  sorry

end chemistry_class_size_l3518_351806


namespace ratio_equation_solution_product_l3518_351855

theorem ratio_equation_solution_product (x : ℝ) : 
  (((x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → x = 0) ∧ 
  (∃ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) := by
  sorry

end ratio_equation_solution_product_l3518_351855


namespace sin_135_degrees_l3518_351800

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_135_degrees_l3518_351800


namespace arithmetic_sequence_problem_l3518_351824

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a₁ : ℚ) (aₙ : ℚ) (n : ℕ) : ℚ := n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_problem (a₁ a₂ a₅ aₙ : ℚ) (n : ℕ) :
  a₁ = 1/3 →
  a₂ + a₅ = 4 →
  aₙ = 33 →
  (∃ d : ℚ, ∀ k : ℕ, arithmetic_sequence a₁ d k = a₁ + (k - 1 : ℚ) * d) →
  n = 50 ∧ sum_arithmetic_sequence a₁ aₙ n = 850 :=
sorry

end arithmetic_sequence_problem_l3518_351824


namespace two_fifths_300_minus_three_fifths_125_l3518_351842

theorem two_fifths_300_minus_three_fifths_125 : 
  (2 : ℚ) / 5 * 300 - (3 : ℚ) / 5 * 125 = 45 := by
  sorry

end two_fifths_300_minus_three_fifths_125_l3518_351842


namespace petya_wins_petya_wins_game_l3518_351850

/-- Represents the game between Petya and Vasya with two boxes of candies. -/
structure CandyGame where
  total_candies : ℕ
  prob_two_caramels : ℝ

/-- Defines the game setup with the given conditions. -/
def game : CandyGame :=
  { total_candies := 25
  , prob_two_caramels := 0.54 }

/-- Theorem stating that Petya has a higher chance of winning the game. -/
theorem petya_wins (g : CandyGame) : g.prob_two_caramels > 0.5 → 
  (1 - g.prob_two_caramels) < 0.5 := by
  sorry

/-- Corollary proving that Petya wins the specific game instance. -/
theorem petya_wins_game : (1 - game.prob_two_caramels) < 0.5 := by
  sorry

end petya_wins_petya_wins_game_l3518_351850


namespace sum_of_even_and_odd_is_odd_l3518_351845

def P : Set ℤ := {x | ∃ k, x = 2 * k}
def Q : Set ℤ := {x | ∃ k, x = 2 * k + 1}
def R : Set ℤ := {x | ∃ k, x = 4 * k + 1}

theorem sum_of_even_and_odd_is_odd (a b : ℤ) (ha : a ∈ P) (hb : b ∈ Q) : 
  a + b ∈ Q := by
  sorry

end sum_of_even_and_odd_is_odd_l3518_351845


namespace leo_laundry_problem_l3518_351847

theorem leo_laundry_problem (total_bill : ℕ) (shirt_cost : ℕ) (trouser_cost : ℕ) (total_shirts : ℕ) :
  total_bill = 140 →
  shirt_cost = 5 →
  trouser_cost = 9 →
  total_shirts = 10 →
  ∃ (num_trousers : ℕ), num_trousers = 10 ∧ total_bill = shirt_cost * total_shirts + trouser_cost * num_trousers :=
by
  sorry

end leo_laundry_problem_l3518_351847


namespace sufficient_implies_necessary_l3518_351820

theorem sufficient_implies_necessary (A B : Prop) :
  (A → B) → (¬B → ¬A) :=
by sorry

end sufficient_implies_necessary_l3518_351820


namespace july_green_tea_price_l3518_351858

/-- Represents the price of tea and coffee in June and July -/
structure PriceData where
  june_price : ℝ
  july_coffee_price : ℝ
  july_tea_price : ℝ

/-- Represents the mixture of tea and coffee -/
structure Mixture where
  tea_quantity : ℝ
  coffee_quantity : ℝ
  total_weight : ℝ
  total_cost : ℝ

/-- Theorem stating the price of green tea in July -/
theorem july_green_tea_price (p : PriceData) (m : Mixture) : 
  p.june_price > 0 ∧ 
  p.july_coffee_price = 2 * p.june_price ∧ 
  p.july_tea_price = 0.1 * p.june_price ∧
  m.tea_quantity = m.coffee_quantity ∧
  m.total_weight = 3 ∧
  m.total_cost = 3.15 ∧
  m.total_cost = m.tea_quantity * p.july_tea_price + m.coffee_quantity * p.july_coffee_price →
  p.july_tea_price = 0.1 := by
sorry


end july_green_tea_price_l3518_351858


namespace original_number_is_nine_l3518_351838

theorem original_number_is_nine (x : ℝ) : (x - 5) / 4 = (x - 4) / 5 → x = 9 := by
  sorry

end original_number_is_nine_l3518_351838


namespace mans_rate_in_still_water_l3518_351889

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate_in_still_water
  (speed_with_stream : ℝ)
  (speed_against_stream : ℝ)
  (h1 : speed_with_stream = 26)
  (h2 : speed_against_stream = 4) :
  (speed_with_stream + speed_against_stream) / 2 = 15 := by
  sorry

end mans_rate_in_still_water_l3518_351889


namespace contradiction_assumption_for_no_real_roots_l3518_351881

theorem contradiction_assumption_for_no_real_roots (a b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + a = 0) ↔ 
  ¬(∀ x : ℝ, x^2 + b*x + a ≠ 0) :=
sorry

end contradiction_assumption_for_no_real_roots_l3518_351881


namespace tangent_perpendicular_range_l3518_351849

/-- The range of a when the tangent lines of two specific curves are perpendicular -/
theorem tangent_perpendicular_range (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 0 (3/2) ∧ 
    ((a * x₀ + a - 1) * (x₀ - 2) = -1)) → 
  a ∈ Set.Icc 1 (3/2) := by
sorry

end tangent_perpendicular_range_l3518_351849


namespace max_value_e_l3518_351851

theorem max_value_e (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 8)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 + e^2 = 16) :
  e ≤ 16/5 ∧ ∃ (a' b' c' d' e' : ℝ), 
    a' + b' + c' + d' + e' = 8 ∧
    a'^2 + b'^2 + c'^2 + d'^2 + e'^2 = 16 ∧
    e' = 16/5 :=
by sorry

end max_value_e_l3518_351851


namespace common_internal_tangent_length_l3518_351856

theorem common_internal_tangent_length
  (center_distance : ℝ)
  (small_radius : ℝ)
  (large_radius : ℝ)
  (h1 : center_distance = 41)
  (h2 : small_radius = 4)
  (h3 : large_radius = 5) :
  Real.sqrt (center_distance^2 - (small_radius + large_radius)^2) = 40 :=
by
  sorry

end common_internal_tangent_length_l3518_351856


namespace rope_cutting_l3518_351885

theorem rope_cutting (l : ℚ) : 
  l > 0 ∧ (1 / l).isInt ∧ (2 / l).isInt → (3 / l) ≠ 8 := by
  sorry

end rope_cutting_l3518_351885


namespace pants_price_calculation_l3518_351868

/-- The price of a T-shirt in dollars -/
def tshirt_price : ℚ := 5

/-- The price of a skirt in dollars -/
def skirt_price : ℚ := 6

/-- The price of a refurbished T-shirt in dollars -/
def refurbished_tshirt_price : ℚ := tshirt_price / 2

/-- The total income from the sales in dollars -/
def total_income : ℚ := 53

/-- The number of T-shirts sold -/
def tshirts_sold : ℕ := 2

/-- The number of pants sold -/
def pants_sold : ℕ := 1

/-- The number of skirts sold -/
def skirts_sold : ℕ := 4

/-- The number of refurbished T-shirts sold -/
def refurbished_tshirts_sold : ℕ := 6

/-- The price of a pair of pants in dollars -/
def pants_price : ℚ := 4

theorem pants_price_calculation :
  pants_price = total_income - 
    (tshirts_sold * tshirt_price + 
     skirts_sold * skirt_price + 
     refurbished_tshirts_sold * refurbished_tshirt_price) :=
by sorry

end pants_price_calculation_l3518_351868


namespace focal_chord_length_l3518_351879

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * Real.sqrt 3 * x

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define a chord passing through the focus
structure FocalChord where
  a : PointOnParabola
  b : PointOnParabola
  passes_through_focus : True  -- We assume this property without specifying the focus

-- Theorem statement
theorem focal_chord_length 
  (ab : FocalChord) 
  (midpoint_x : ab.a.x + ab.b.x = 4) : 
  Real.sqrt ((ab.a.x - ab.b.x)^2 + (ab.a.y - ab.b.y)^2) = 4 + 2 * Real.sqrt 3 := by
  sorry

end focal_chord_length_l3518_351879


namespace number_product_l3518_351817

theorem number_product (x : ℝ) : x - 7 = 9 → 5 * x = 80 := by
  sorry

end number_product_l3518_351817


namespace rational_solution_quadratic_l3518_351860

theorem rational_solution_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 24 * x + 4 * k = 0) ↔ k = 6 := by
  sorry

end rational_solution_quadratic_l3518_351860


namespace point_translation_second_quadrant_l3518_351893

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point by a given vector -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

/-- Check if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The main theorem -/
theorem point_translation_second_quadrant (m n : ℝ) :
  let A : Point := { x := m, y := n }
  let A' : Point := translate A 2 3
  isInSecondQuadrant A' → m < -2 ∧ n > -3 := by
  sorry

end point_translation_second_quadrant_l3518_351893


namespace vector_dot_product_l3518_351874

theorem vector_dot_product (a b : ℝ × ℝ) : 
  (Real.sqrt 2 : ℝ) = Real.sqrt (a.1 ^ 2 + a.2 ^ 2) →
  2 = Real.sqrt (b.1 ^ 2 + b.2 ^ 2) →
  (3 * Real.pi / 4 : ℝ) = Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1 ^ 2 + a.2 ^ 2) * Real.sqrt (b.1 ^ 2 + b.2 ^ 2))) →
  (a.1 * (a.1 - 2 * b.1) + a.2 * (a.2 - 2 * b.2) : ℝ) = 6 := by
sorry

end vector_dot_product_l3518_351874


namespace quadratic_roots_imply_m_value_l3518_351834

/-- If the roots of the quadratic 10x^2 - 6x + m are (3 ± i√191)/10, then m = 227/40 -/
theorem quadratic_roots_imply_m_value (m : ℝ) : 
  (∃ x : ℂ, x^2 * 10 - x * 6 + m = 0 ∧ x = (3 + Complex.I * Real.sqrt 191) / 10) →
  m = 227 / 40 := by
  sorry

end quadratic_roots_imply_m_value_l3518_351834


namespace fish_difference_l3518_351830

theorem fish_difference (goldfish : ℕ) (angelfish : ℕ) (guppies : ℕ) : 
  goldfish = 8 →
  guppies = 2 * angelfish →
  goldfish + angelfish + guppies = 44 →
  angelfish - goldfish = 4 := by
sorry

end fish_difference_l3518_351830


namespace sum_of_powers_of_i_equals_zero_l3518_351831

theorem sum_of_powers_of_i_equals_zero (i : ℂ) (hi : i^2 = -1) :
  i^1234 + i^1235 + i^1236 + i^1237 = 0 := by
  sorry

end sum_of_powers_of_i_equals_zero_l3518_351831


namespace line_equation_through_intersection_and_parallel_l3518_351853

/-- Given two lines in the plane and a third line parallel to one of them,
    this theorem proves the equation of the third line. -/
theorem line_equation_through_intersection_and_parallel
  (l₁ l₂ l₃ l : Real → Real → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ 3 * x + 5 * y - 4 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ 6 * x - y + 3 = 0)
  (h₃ : ∀ x y, l₃ x y ↔ 2 * x + 3 * y + 5 = 0)
  (h_intersect : ∃ x y, l₁ x y ∧ l₂ x y ∧ l x y)
  (h_parallel : ∃ k ≠ 0, ∀ x y, l x y ↔ 2 * k * x + 3 * k * y + (k * 5 + c) = 0) :
  ∀ x y, l x y ↔ 6 * x + 9 * y - 7 = 0 := by
sorry

end line_equation_through_intersection_and_parallel_l3518_351853


namespace gcf_lcm_product_4_12_l3518_351812

theorem gcf_lcm_product_4_12 : 
  (Nat.gcd 4 12) * (Nat.lcm 4 12) = 48 := by
  sorry

end gcf_lcm_product_4_12_l3518_351812


namespace willow_football_time_l3518_351857

/-- Proves that Willow played football for 60 minutes given the conditions -/
theorem willow_football_time :
  ∀ (total_time basketball_time football_time : ℕ),
  total_time = 120 →
  basketball_time = 60 →
  total_time = basketball_time + football_time →
  football_time = 60 :=
by
  sorry

end willow_football_time_l3518_351857


namespace inverse_proportional_solution_l3518_351814

def inverse_proportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportional_solution (x y : ℝ) 
  (h1 : inverse_proportional x y) 
  (h2 : x + y = 30) 
  (h3 : x - y = 10) : 
  (∃ y' : ℝ, inverse_proportional 8 y' ∧ y' = 25) :=
by sorry

end inverse_proportional_solution_l3518_351814


namespace unique_root_quadratic_l3518_351873

theorem unique_root_quadratic (b c : ℝ) : 
  (∃! x : ℝ, x^2 + b*x + c = 0) → 
  (b = c + 1) → 
  c = 1 :=
by
  sorry

#check unique_root_quadratic

end unique_root_quadratic_l3518_351873


namespace equation_solution_range_l3518_351898

theorem equation_solution_range (a : ℝ) : 
  (∃ x : ℝ, 9^x + (a+4)*3^x + 4 = 0) ↔ a ≤ -8 := by sorry

end equation_solution_range_l3518_351898


namespace simplest_fraction_of_0_63575_l3518_351876

theorem simplest_fraction_of_0_63575 :
  ∃ (a b : ℕ+), (a.val : ℚ) / b.val = 63575 / 100000 ∧
  ∀ (c d : ℕ+), (c.val : ℚ) / d.val = 63575 / 100000 → a.val ≤ c.val ∧ b.val ≤ d.val →
  a = 2543 ∧ b = 4000 := by
sorry

end simplest_fraction_of_0_63575_l3518_351876


namespace fruit_basket_count_l3518_351809

def total_fruits (mangoes pears pawpaws kiwis lemons : ℕ) : ℕ :=
  mangoes + pears + pawpaws + kiwis + lemons

theorem fruit_basket_count : 
  ∀ (kiwis : ℕ),
  kiwis = 9 →
  total_fruits 18 10 12 kiwis 9 = 58 := by
sorry

end fruit_basket_count_l3518_351809


namespace ratio_to_eleven_l3518_351891

theorem ratio_to_eleven : ∃ x : ℚ, (5 : ℚ) / 1 = x / 11 ∧ x = 55 := by
  sorry

end ratio_to_eleven_l3518_351891


namespace inequality_solution_range_l3518_351832

theorem inequality_solution_range (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 4, x^2 + a*x - 2 < 0) ↔ a < 1 := by
  sorry

end inequality_solution_range_l3518_351832


namespace flight_cost_a_to_b_l3518_351801

/-- Represents the cost of a flight between two cities -/
structure FlightCost where
  bookingFee : ℝ
  ratePerKm : ℝ

/-- Calculates the total cost of a flight -/
def calculateFlightCost (distance : ℝ) (cost : FlightCost) : ℝ :=
  cost.bookingFee + cost.ratePerKm * distance

/-- The problem statement -/
theorem flight_cost_a_to_b :
  let distanceAB : ℝ := 3500
  let flightCost : FlightCost := { bookingFee := 120, ratePerKm := 0.12 }
  calculateFlightCost distanceAB flightCost = 540 := by
  sorry


end flight_cost_a_to_b_l3518_351801


namespace money_division_l3518_351865

theorem money_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 406)
  (h_a : a = b / 2)
  (h_b : b = c / 2)
  (h_sum : a + b + c = total) : c = 232 := by
  sorry

end money_division_l3518_351865


namespace student_competition_numbers_l3518_351884

theorem student_competition_numbers (n : ℕ) : 
  (100 < n ∧ n < 200) ∧ 
  (∃ k : ℕ, n = 4 * k - 2) ∧
  (∃ l : ℕ, n = 5 * l - 3) ∧
  (∃ m : ℕ, n = 6 * m - 4) →
  n = 122 ∨ n = 182 := by
sorry

end student_competition_numbers_l3518_351884


namespace hostel_rate_is_15_l3518_351862

/-- Represents the lodging problem for Jimmy's vacation --/
def lodging_problem (hostel_rate : ℚ) : Prop :=
  let hostel_nights : ℕ := 3
  let cabin_nights : ℕ := 2
  let cabin_rate : ℚ := 45
  let cabin_people : ℕ := 3
  let total_cost : ℚ := 75
  (hostel_nights : ℚ) * hostel_rate + 
    (cabin_nights : ℚ) * (cabin_rate / cabin_people) = total_cost

/-- Theorem stating that the hostel rate is $15 per night --/
theorem hostel_rate_is_15 : 
  lodging_problem 15 := by sorry

end hostel_rate_is_15_l3518_351862


namespace point_position_l3518_351810

def line (x y : ℝ) := x + 2 * y = 2

def point_below_left (P : ℝ × ℝ) : Prop :=
  P.1 + 2 * P.2 < 2

theorem point_position :
  let P : ℝ × ℝ := (1/12, 33/36)
  point_below_left P := by sorry

end point_position_l3518_351810


namespace inscribed_quadrilateral_slope_l3518_351811

/-- Given a quadrilateral ABCD inscribed in an ellipse, with three sides AB, BC, CD parallel to fixed directions,
    the slope of the fourth side DA is determined by the slopes of the other three sides and the ellipse parameters. -/
theorem inscribed_quadrilateral_slope (a b : ℝ) (m₁ m₂ m₃ : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  ∃ m : ℝ, m = (b^2 * (m₁ + m₃ - m₂) + a^2 * m₁ * m₂ * m₃) / (b^2 + a^2 * (m₁ * m₂ + m₂ * m₃ - m₁ * m₃)) :=
by sorry

end inscribed_quadrilateral_slope_l3518_351811


namespace max_sum_ITEST_l3518_351818

theorem max_sum_ITEST (I T E S : ℕ+) : 
  I ≠ T ∧ I ≠ E ∧ I ≠ S ∧ T ≠ E ∧ T ≠ S ∧ E ≠ S →
  I * T * E * S * T = 2006 →
  (∀ (I' T' E' S' : ℕ+), 
    I' ≠ T' ∧ I' ≠ E' ∧ I' ≠ S' ∧ T' ≠ E' ∧ T' ≠ S' ∧ E' ≠ S' →
    I' * T' * E' * S' * T' = 2006 →
    I + T + E + S + T + 2006 ≥ I' + T' + E' + S' + T' + 2006) →
  I + T + E + S + T + 2006 = 2086 := by
sorry

end max_sum_ITEST_l3518_351818


namespace tan_alpha_value_l3518_351895

theorem tan_alpha_value (α : Real) (h1 : π/2 < α ∧ α < π) 
  (h2 : Real.sin (α + π/4) = Real.sqrt 2 / 10) : Real.tan α = -4/3 := by
  sorry

end tan_alpha_value_l3518_351895


namespace problem_pyramid_rows_l3518_351883

/-- Represents a pyramid display of cans -/
structure CanPyramid where
  topRowCans : ℕ
  rowIncrement : ℕ
  totalCans : ℕ

/-- Calculates the number of rows in a can pyramid -/
def numberOfRows (p : CanPyramid) : ℕ :=
  sorry

/-- The specific can pyramid from the problem -/
def problemPyramid : CanPyramid :=
  { topRowCans := 3
  , rowIncrement := 3
  , totalCans := 225 }

/-- Theorem stating that the number of rows in the problem pyramid is 12 -/
theorem problem_pyramid_rows :
  numberOfRows problemPyramid = 12 := by
  sorry

end problem_pyramid_rows_l3518_351883


namespace probability_is_correct_l3518_351890

/-- Represents the total number of cards -/
def t : ℕ := 93

/-- Represents the number of cards with blue dinosaurs -/
def blue_dinosaurs : ℕ := 16

/-- Represents the number of cards with green robots -/
def green_robots : ℕ := 14

/-- Represents the number of cards with blue robots -/
def blue_robots : ℕ := 36

/-- Represents the number of cards with green dinosaurs -/
def green_dinosaurs : ℕ := t - (blue_dinosaurs + green_robots + blue_robots)

/-- The probability of choosing a card with either a green dinosaur or a blue robot -/
def probability : ℚ := (green_dinosaurs + blue_robots : ℚ) / t

theorem probability_is_correct : probability = 21 / 31 := by
  sorry

end probability_is_correct_l3518_351890


namespace f_of_4_equals_9_l3518_351828

/-- The function f is defined as f(x) = x^2 - 2x + 1 for all x. -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- Theorem: The value of f(4) is 9. -/
theorem f_of_4_equals_9 : f 4 = 9 := by
  sorry

end f_of_4_equals_9_l3518_351828


namespace parallelogram_not_always_axisymmetric_and_centrally_symmetric_l3518_351807

-- Define a parallelogram
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ
  is_parallelogram : ∀ (i : Fin 4), 
    vertices i - vertices ((i + 1) % 4) = vertices ((i + 2) % 4) - vertices ((i + 3) % 4)

-- Define an axisymmetric figure
def IsAxisymmetric (vertices : Fin 4 → ℝ × ℝ) : Prop :=
  ∃ (axis : ℝ × ℝ → ℝ × ℝ), ∀ (i : Fin 4), 
    axis (vertices i) = vertices ((4 - i) % 4)

-- Define a centrally symmetric figure
def IsCentrallySymmetric (vertices : Fin 4 → ℝ × ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), ∀ (i : Fin 4), 
    vertices i - center = center - vertices ((i + 2) % 4)

-- Theorem statement
theorem parallelogram_not_always_axisymmetric_and_centrally_symmetric :
  ¬(∀ (p : Parallelogram), IsAxisymmetric p.vertices ∧ IsCentrallySymmetric p.vertices) :=
sorry

end parallelogram_not_always_axisymmetric_and_centrally_symmetric_l3518_351807


namespace imaginary_part_of_complex_fraction_l3518_351848

theorem imaginary_part_of_complex_fraction : Complex.im (2 * Complex.I / (1 - Complex.I)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l3518_351848


namespace video_game_spending_l3518_351875

theorem video_game_spending (weekly_allowance : ℝ) (weeks : ℕ) 
  (video_game_cost : ℝ) (book_fraction : ℝ) (remaining : ℝ) :
  weekly_allowance = 10 →
  weeks = 4 →
  book_fraction = 1/4 →
  remaining = 15 →
  video_game_cost > 0 →
  video_game_cost < weekly_allowance * weeks →
  remaining = weekly_allowance * weeks - video_game_cost - 
    (weekly_allowance * weeks - video_game_cost) * book_fraction →
  video_game_cost / (weekly_allowance * weeks) = 1/2 := by
sorry

end video_game_spending_l3518_351875


namespace gcd_of_45139_34481_4003_l3518_351863

theorem gcd_of_45139_34481_4003 : Nat.gcd 45139 (Nat.gcd 34481 4003) = 1 := by
  sorry

end gcd_of_45139_34481_4003_l3518_351863


namespace brianna_books_to_reread_l3518_351870

/-- The number of books Brianna reads per month -/
def books_per_month : ℕ := 2

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of new books Brianna was given as a gift -/
def books_gifted : ℕ := 6

/-- The number of new books Brianna bought -/
def books_bought : ℕ := 8

/-- The number of new books Brianna plans to borrow from the library -/
def books_borrowed : ℕ := books_bought - 2

/-- The total number of books Brianna needs for the year -/
def total_books_needed : ℕ := books_per_month * months_in_year

/-- The total number of new books Brianna will have -/
def total_new_books : ℕ := books_gifted + books_bought + books_borrowed

/-- The number of old books Brianna needs to reread -/
def old_books_to_reread : ℕ := total_books_needed - total_new_books

theorem brianna_books_to_reread : old_books_to_reread = 4 := by
  sorry

end brianna_books_to_reread_l3518_351870
