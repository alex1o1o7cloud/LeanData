import Mathlib

namespace contestant_final_score_l519_51919

/-- Calculates the final score of a contestant given their individual scores and weightings -/
def final_score (etiquette_score language_score behavior_score : ℝ)
  (etiquette_weight language_weight behavior_weight : ℝ) : ℝ :=
  etiquette_score * etiquette_weight +
  language_score * language_weight +
  behavior_score * behavior_weight

/-- Theorem stating that the contestant's final score is 89 points -/
theorem contestant_final_score :
  final_score 95 92 80 0.4 0.25 0.35 = 89 := by
  sorry

end contestant_final_score_l519_51919


namespace sphere_hemisphere_cone_volume_ratio_l519_51915

/-- The ratio of the volume of a sphere to the combined volume of a hemisphere and a cone -/
theorem sphere_hemisphere_cone_volume_ratio (r : ℝ) (hr : r > 0) : 
  (4 / 3 * π * r^3) / ((1 / 2 * 4 / 3 * π * (3 * r)^3) + (1 / 3 * π * r^2 * (2 * r))) = 1 / 14 := by
  sorry

#check sphere_hemisphere_cone_volume_ratio

end sphere_hemisphere_cone_volume_ratio_l519_51915


namespace ceiling_fraction_evaluation_l519_51960

theorem ceiling_fraction_evaluation : 
  (⌈(19 / 8 : ℚ) - ⌈(35 / 19 : ℚ)⌉⌉ : ℚ) / 
  (⌈(35 / 8 : ℚ) + ⌈(8 * 19 / 35 : ℚ)⌉⌉ : ℚ) = 1 / 10 := by sorry

end ceiling_fraction_evaluation_l519_51960


namespace complement_of_M_in_U_l519_51934

def U : Set ℤ := {1, 2, 3, 4, 5, 6, 7}

def M : Set ℤ := {x | x^2 - 6*x + 5 ≤ 0 ∧ x ∈ U}

theorem complement_of_M_in_U :
  U \ M = {6, 7} := by sorry

end complement_of_M_in_U_l519_51934


namespace certain_number_problem_l519_51906

theorem certain_number_problem (x : ℝ) : ((x + 20) * 2) / 2 - 2 = 88 / 2 ↔ x = 26 := by
  sorry

end certain_number_problem_l519_51906


namespace factorization_1_factorization_2_l519_51912

-- Problem 1
theorem factorization_1 (a x y : ℝ) : a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) := by
  sorry

-- Problem 2
theorem factorization_2 (x : ℝ) : x^3 - 8 * x^2 + 16 * x = x * (x - 4)^2 := by
  sorry

end factorization_1_factorization_2_l519_51912


namespace parallel_lines_m_value_l519_51995

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, m1 * x + y + b1 = 0 ↔ m2 * x + y + b2 = 0) ↔ m1 = m2

/-- The problem statement -/
theorem parallel_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, 2 * x + (m + 1) * y + 4 = 0 ↔ m * x + 3 * y + 4 = 0) →
  m = -3 :=
by sorry

end parallel_lines_m_value_l519_51995


namespace polynomial_division_remainder_l519_51933

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X : Polynomial ℝ)^4 + (X : Polynomial ℝ)^2 - 5 = 
  (X^2 - 3) * q + (4 * X^2 - 5) := by sorry

end polynomial_division_remainder_l519_51933


namespace max_moves_card_game_l519_51983

/-- Represents the state of the cards as a natural number -/
def initial_state : Nat := 43690

/-- Represents a valid move in the game -/
def is_valid_move (n : Nat) : Prop :=
  ∃ k, 0 < k ∧ k ≤ 16 ∧ n.mod (2^k) = 2^(k-1)

/-- The game ends when no valid move can be made -/
def game_ended (n : Nat) : Prop :=
  ¬∃ m, is_valid_move n ∧ m < n

/-- Theorem stating the maximum number of moves in the game -/
theorem max_moves_card_game :
  ∃ moves : Nat, moves = initial_state ∧
  (∀ n, n > moves → ¬∃ seq : Nat → Nat, seq 0 = initial_state ∧
    (∀ i < n, is_valid_move (seq i) ∧ seq (i+1) < seq i) ∧
    game_ended (seq n)) :=
sorry

end max_moves_card_game_l519_51983


namespace shop_earnings_l519_51955

theorem shop_earnings : 
  let cola_price : ℚ := 3
  let juice_price : ℚ := 3/2
  let water_price : ℚ := 1
  let cola_sold : ℕ := 15
  let juice_sold : ℕ := 12
  let water_sold : ℕ := 25
  cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold = 88
  := by sorry

end shop_earnings_l519_51955


namespace sum_of_solutions_is_zero_l519_51985

theorem sum_of_solutions_is_zero : 
  ∃ (x₁ x₂ : ℝ), (8 * x₁) / 40 = 7 / x₁ ∧ 
                 (8 * x₂) / 40 = 7 / x₂ ∧ 
                 x₁ + x₂ = 0 ∧
                 ∀ (y : ℝ), (8 * y) / 40 = 7 / y → y = x₁ ∨ y = x₂ := by
  sorry

end sum_of_solutions_is_zero_l519_51985


namespace convex_polygon_30_sides_diagonals_l519_51982

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 30 sides has 202 diagonals -/
theorem convex_polygon_30_sides_diagonals :
  num_diagonals 30 = 202 := by
  sorry

end convex_polygon_30_sides_diagonals_l519_51982


namespace exists_2x2_square_after_removal_l519_51923

/-- Represents a cell in the grid -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents a two-cell rectangle (domino) -/
structure Domino where
  cell1 : Cell
  cell2 : Cell

/-- The grid size -/
def gridSize : Nat := 100

/-- The number of dominoes removed -/
def removedDominoes : Nat := 1950

/-- Function to check if a cell is within the grid -/
def isValidCell (c : Cell) : Prop :=
  c.row < gridSize ∧ c.col < gridSize

/-- Function to check if two cells are adjacent -/
def areAdjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ (c1.col + 1 = c2.col ∨ c2.col + 1 = c1.col)) ∨
  (c1.col = c2.col ∧ (c1.row + 1 = c2.row ∨ c2.row + 1 = c1.row))

/-- Function to check if a domino is valid -/
def isValidDomino (d : Domino) : Prop :=
  isValidCell d.cell1 ∧ isValidCell d.cell2 ∧ areAdjacent d.cell1 d.cell2

/-- Theorem: After removing 1950 dominoes, there exists a 2x2 square in the remaining cells -/
theorem exists_2x2_square_after_removal 
  (removed : Finset Domino) 
  (h_removed : removed.card = removedDominoes) 
  (h_valid : ∀ d ∈ removed, isValidDomino d) :
  ∃ (c : Cell), isValidCell c ∧ 
    isValidCell { row := c.row, col := c.col + 1 } ∧ 
    isValidCell { row := c.row + 1, col := c.col } ∧ 
    isValidCell { row := c.row + 1, col := c.col + 1 } ∧
    (∀ d ∈ removed, d.cell1 ≠ c ∧ d.cell2 ≠ c) ∧
    (∀ d ∈ removed, d.cell1 ≠ { row := c.row, col := c.col + 1 } ∧ d.cell2 ≠ { row := c.row, col := c.col + 1 }) ∧
    (∀ d ∈ removed, d.cell1 ≠ { row := c.row + 1, col := c.col } ∧ d.cell2 ≠ { row := c.row + 1, col := c.col }) ∧
    (∀ d ∈ removed, d.cell1 ≠ { row := c.row + 1, col := c.col + 1 } ∧ d.cell2 ≠ { row := c.row + 1, col := c.col + 1 }) :=
by sorry

end exists_2x2_square_after_removal_l519_51923


namespace trapezoid_existence_l519_51993

-- Define the trapezoid structure
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ

-- Define the existence theorem
theorem trapezoid_existence (a b c α β : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) : 
  ∃ t : Trapezoid, 
    t.a = a ∧ t.b = b ∧ t.c = c ∧ t.α = α ∧ t.β = β :=
sorry


end trapezoid_existence_l519_51993


namespace wire_length_proof_l519_51921

theorem wire_length_proof (shorter_piece longer_piece total_length : ℝ) : 
  shorter_piece = 20 →
  shorter_piece = (2 / 7) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 90 := by
sorry

end wire_length_proof_l519_51921


namespace right_triangle_consecutive_even_sides_l519_51968

theorem right_triangle_consecutive_even_sides (a b c : ℕ) : 
  (∃ x : ℕ, a = x - 2 ∧ b = x ∧ c = x + 2) →  -- sides are consecutive even numbers
  (a^2 + b^2 = c^2) →                        -- right-angled triangle
  c = 10                                     -- hypotenuse length is 10
  := by sorry

end right_triangle_consecutive_even_sides_l519_51968


namespace min_value_of_f_l519_51978

theorem min_value_of_f (x : ℝ) : 1 / Real.sqrt (x^2 + 2) + Real.sqrt (x^2 + 2) ≥ 3 * Real.sqrt 2 / 2 := by
  sorry

end min_value_of_f_l519_51978


namespace simplify_fraction_l519_51930

theorem simplify_fraction : (222 : ℚ) / 8888 * 22 = 1 / 2 := by sorry

end simplify_fraction_l519_51930


namespace sock_ratio_l519_51907

theorem sock_ratio (total : ℕ) (blue : ℕ) (h1 : total = 180) (h2 : blue = 60) :
  (total - blue) / total = 2 / 3 := by
sorry

end sock_ratio_l519_51907


namespace game_winning_strategy_l519_51911

def game_move (k : ℕ) : Set ℕ := {k + 1, 2 * k}

def is_winning_position (n : ℕ) : Prop :=
  ∃ (k c : ℕ), n = 2^(2*k+1) + 2*c ∧ c < 2^k

theorem game_winning_strategy (n : ℕ) (h : n > 1) :
  (∀ k, k ∈ game_move 2 → k ≤ n) →
  (is_winning_position n ↔ 
    ∃ (strategy : ℕ → ℕ), 
      (∀ m, m < n → strategy m ∈ game_move m) ∧
      (∀ m, m < n → strategy (strategy m) > n)) :=
sorry

end game_winning_strategy_l519_51911


namespace odd_heads_probability_l519_51932

/-- The probability of getting heads for the kth coin -/
def p (k : ℕ) : ℚ := 1 / (2 * k + 1)

/-- The probability of getting an odd number of heads when tossing n biased coins -/
def odd_heads_prob (n : ℕ) : ℚ := n / (2 * n + 1)

/-- Theorem stating that the probability of getting an odd number of heads
    when tossing n biased coins is n/(2n+1), where the kth coin has
    probability 1/(2k+1) of falling heads -/
theorem odd_heads_probability (n : ℕ) :
  (∀ k, k ≤ n → p k = 1 / (2 * k + 1)) →
  odd_heads_prob n = n / (2 * n + 1) := by
  sorry

end odd_heads_probability_l519_51932


namespace unique_solution_E_l519_51972

/-- Definition of the function E -/
def E (a b c : ℝ) : ℝ := a * b^2 + c

/-- Theorem stating that 3/8 is the unique solution to E(a, 3, 12) = E(a, 5, 6) -/
theorem unique_solution_E :
  ∃! a : ℝ, E a 3 12 = E a 5 6 ∧ a = 3/8 := by
  sorry

end unique_solution_E_l519_51972


namespace seventh_term_of_geometric_sequence_l519_51942

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem seventh_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_positive : ∀ n, a n > 0) 
  (h_fifth : a 5 = 16) 
  (h_ninth : a 9 = 4) : 
  a 7 = 8 := by
sorry

end seventh_term_of_geometric_sequence_l519_51942


namespace laptop_cost_ratio_l519_51950

theorem laptop_cost_ratio : 
  ∀ (first_laptop_cost second_laptop_cost : ℝ),
    first_laptop_cost = 500 →
    first_laptop_cost + second_laptop_cost = 2000 →
    second_laptop_cost / first_laptop_cost = 3 := by
  sorry

end laptop_cost_ratio_l519_51950


namespace cube_root_equality_l519_51969

theorem cube_root_equality (a b : ℝ) :
  (a ^ (1/3 : ℝ) = -(b ^ (1/3 : ℝ))) → a = -b := by
  sorry

end cube_root_equality_l519_51969


namespace twentieth_triangular_number_l519_51945

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 20th triangular number is 210 -/
theorem twentieth_triangular_number : triangular_number 20 = 210 := by
  sorry

end twentieth_triangular_number_l519_51945


namespace product_sum_theorem_l519_51956

theorem product_sum_theorem :
  ∀ (a b c d : ℝ),
  (∀ x : ℝ, (5 * x^2 - 3 * x + 7) * (9 - 4 * x) = a * x^3 + b * x^2 + c * x + d) →
  8 * a + 4 * b + 2 * c + d = -29 := by
sorry

end product_sum_theorem_l519_51956


namespace ratio_problem_l519_51974

theorem ratio_problem (x : ℝ) : 
  (5 : ℝ) * x = 60 → x = 12 := by
sorry

end ratio_problem_l519_51974


namespace valid_colorings_2x9_board_l519_51947

/-- Represents the number of columns in the board -/
def n : ℕ := 9

/-- Represents the number of colors available -/
def num_colors : ℕ := 3

/-- Represents the number of ways to color the first column -/
def first_column_colorings : ℕ := num_colors * (num_colors - 1)

/-- Represents the number of ways to color each subsequent column -/
def subsequent_column_colorings : ℕ := num_colors - 1

/-- Theorem stating the number of valid colorings for a 2 × 9 board -/
theorem valid_colorings_2x9_board :
  first_column_colorings * subsequent_column_colorings^(n - 1) = 39366 := by
  sorry

end valid_colorings_2x9_board_l519_51947


namespace candy_jar_problem_l519_51903

theorem candy_jar_problem (total : ℕ) (red : ℕ) (blue : ℕ) : 
  total = 3409 → red = 145 → blue = total - red → blue = 3264 := by
  sorry

end candy_jar_problem_l519_51903


namespace x_value_proof_l519_51929

theorem x_value_proof (x : ℝ) (h : x^2 * 8^3 / 256 = 450) : x = 15 ∨ x = -15 := by
  sorry

end x_value_proof_l519_51929


namespace additional_weight_needed_l519_51957

/-- Calculates the additional weight needed to open the cave doors -/
theorem additional_weight_needed 
  (set1_weight : ℝ) 
  (set1_count : ℕ) 
  (set2_weight : ℝ) 
  (set2_count : ℕ) 
  (switch_weight : ℝ) 
  (total_needed : ℝ) 
  (large_rock_kg : ℝ) 
  (kg_to_lbs : ℝ) 
  (h1 : set1_weight = 60) 
  (h2 : set1_count = 3) 
  (h3 : set2_weight = 42) 
  (h4 : set2_count = 5) 
  (h5 : switch_weight = 234) 
  (h6 : total_needed = 712) 
  (h7 : large_rock_kg = 12) 
  (h8 : kg_to_lbs = 2.2) : 
  total_needed - (switch_weight + set1_weight * set1_count + set2_weight * set2_count + large_rock_kg * kg_to_lbs) = 61.6 := by
  sorry

#check additional_weight_needed

end additional_weight_needed_l519_51957


namespace line_parallel_from_plane_parallel_l519_51952

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Define the parallelism relation for lines and planes
variable (parallel : Line → Line → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_from_plane_parallel
  (a b : Line) (α β γ δ : Plane)
  (h_distinct_lines : a ≠ b)
  (h_distinct_planes : α ≠ β ∧ α ≠ γ ∧ α ≠ δ ∧ β ≠ γ ∧ β ≠ δ ∧ γ ≠ δ)
  (h_intersect_ab : intersect α β = a)
  (h_intersect_gd : intersect γ δ = b)
  (h_parallel_ag : planeParallel α γ)
  (h_parallel_bd : planeParallel β δ) :
  parallel a b :=
sorry

end line_parallel_from_plane_parallel_l519_51952


namespace function_extrema_l519_51951

open Real

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x - log x

theorem function_extrema (a b : ℝ) :
  (a = -1 ∧ b = 3 →
    (∃ (max min : ℝ),
      (∀ x ∈ Set.Icc (1/2) 2, f a b x ≤ max) ∧
      (∃ x ∈ Set.Icc (1/2) 2, f a b x = max) ∧
      (∀ x ∈ Set.Icc (1/2) 2, f a b x ≥ min) ∧
      (∃ x ∈ Set.Icc (1/2) 2, f a b x = min) ∧
      max = 2 ∧
      min = log 2 + 5/4)) ∧
  (a = 0 →
    (∃! b : ℝ,
      b > 0 ∧
      (∃ min : ℝ,
        (∀ x ∈ Set.Ioo 0 (exp 1), f a b x ≥ min) ∧
        (∃ x ∈ Set.Ioo 0 (exp 1), f a b x = min) ∧
        min = 3) ∧
      b = exp 2)) :=
sorry

end function_extrema_l519_51951


namespace polynomial_factorization_l519_51988

theorem polynomial_factorization (x y : ℝ) : 
  (x - 2*y) * (x - 2*y + 1) = x^2 - 4*x*y - 2*y + x + 4*y^2 := by
  sorry

end polynomial_factorization_l519_51988


namespace ladder_height_proof_l519_51937

def ceiling_height : ℝ := 300
def fixture_below_ceiling : ℝ := 15
def alice_height : ℝ := 170
def alice_normal_reach : ℝ := 55
def extra_reach_needed : ℝ := 5

theorem ladder_height_proof :
  let fixture_height := ceiling_height - fixture_below_ceiling
  let total_reach_needed := fixture_height
  let alice_max_reach := alice_height + alice_normal_reach + extra_reach_needed
  let ladder_height := total_reach_needed - alice_max_reach
  ladder_height = 60 := by sorry

end ladder_height_proof_l519_51937


namespace cost_comparison_l519_51931

/-- The price of a suit in yuan -/
def suit_price : ℕ := 1000

/-- The price of a tie in yuan -/
def tie_price : ℕ := 200

/-- The number of suits to be purchased -/
def num_suits : ℕ := 20

/-- The discount rate for Option 2 -/
def discount_rate : ℚ := 9/10

/-- The cost calculation for Option 1 -/
def option1_cost (x : ℕ) : ℕ := 
  num_suits * suit_price + (x - num_suits) * tie_price

/-- The cost calculation for Option 2 -/
def option2_cost (x : ℕ) : ℚ := 
  discount_rate * (num_suits * suit_price + x * tie_price)

theorem cost_comparison (x : ℕ) (h : x > num_suits) : 
  option1_cost x = 200 * x + 16000 ∧ 
  option2_cost x = 180 * x + 18000 := by
  sorry

#check cost_comparison

end cost_comparison_l519_51931


namespace game_a_vs_game_b_l519_51953

def coin_prob_heads : ℚ := 2/3
def coin_prob_tails : ℚ := 1/3

def game_a_win (p : ℚ) : ℚ := p^4 + (1-p)^4

def game_b_win (p : ℚ) : ℚ := p^3 * (1-p) + (1-p)^3 * p

theorem game_a_vs_game_b :
  game_a_win coin_prob_heads - game_b_win coin_prob_heads = 7/81 :=
by sorry

end game_a_vs_game_b_l519_51953


namespace semicircle_curve_length_l519_51925

open Real

/-- The length of the curve traced by point D in a semicircle configuration --/
theorem semicircle_curve_length (k : ℝ) (h : k > 0) :
  ∃ (curve_length : ℝ),
    (∀ (θ : ℝ), 0 ≤ θ ∧ θ ≤ π / 2 →
      let C : ℝ × ℝ := (cos (2 * θ), sin (2 * θ))
      let D : ℝ × ℝ := (cos (2 * θ) + k * sin (2 * θ), sin (2 * θ) + k * (1 - cos (2 * θ)))
      (D.1 ^ 2 + (D.2 - k) ^ 2 = 1 + k ^ 2) ∧
      (k * D.1 + D.2 ≥ k)) →
    curve_length = π * sqrt (1 + k ^ 2) :=
by sorry

end semicircle_curve_length_l519_51925


namespace simplify_expression_1_simplify_expression_2_l519_51990

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  2 * x^2 - 3 * y^2 + 6 * x - x^2 + 3 * y^2 = x^2 + 6 * x :=
by sorry

-- Problem 2
theorem simplify_expression_2 (m : ℝ) :
  4 * m^2 + 1 + 2 * m - 3 * (2 + m - m^2) = 7 * m^2 - m - 5 :=
by sorry

end simplify_expression_1_simplify_expression_2_l519_51990


namespace square_differences_sum_l519_51998

theorem square_differences_sum : 1010^2 - 990^2 - 1005^2 + 995^2 - 1002^2 + 998^2 = 28000 := by
  sorry

end square_differences_sum_l519_51998


namespace number_of_bowls_l519_51946

theorem number_of_bowls (n : ℕ) : n > 0 → (96 : ℝ) / n = 6 → n = 16 := by
  sorry

end number_of_bowls_l519_51946


namespace reflection_path_exists_l519_51973

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a path from A to B with two reflections in a triangle -/
structure ReflectionPath (t : Triangle) where
  P : Point -- Point of reflection on side BC
  Q : Point -- Point of reflection on side CA

/-- Angle at vertex C of a triangle -/
def angle_C (t : Triangle) : ℝ := sorry

/-- Theorem stating the condition for the existence of a reflection path -/
theorem reflection_path_exists (t : Triangle) : 
  (∃ path : ReflectionPath t, True) ↔ (π/4 < angle_C t ∧ angle_C t < π/3) := by sorry

end reflection_path_exists_l519_51973


namespace det_example_and_cube_diff_sum_of_cubes_given_conditions_complex_det_sum_given_conditions_l519_51944

-- Define the determinant operation
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Statement 1
theorem det_example_and_cube_diff : 
  det 5 4 8 9 = 13 ∧ ∀ a : ℝ, a^3 - 3*a^2 + 3*a - 1 = (a - 1)^3 := by sorry

-- Statement 2
theorem sum_of_cubes_given_conditions :
  ∀ x y : ℝ, x + y = 3 → x * y = 1 → x^3 + y^3 = 18 := by sorry

-- Statement 3
theorem complex_det_sum_given_conditions :
  ∀ x m n : ℝ, m = x - 1 → n = x + 2 → m * n = 5 →
  det m (3*m^2 + n^2) n (m^2 + 3*n^2) + det (m + n) (-2*n) n (m - n) = -8 := by sorry

end det_example_and_cube_diff_sum_of_cubes_given_conditions_complex_det_sum_given_conditions_l519_51944


namespace inscribed_triangle_sides_l519_51910

/-- A triangle with an inscribed circle -/
structure InscribedTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first side of the triangle -/
  a : ℝ
  /-- The length of the second side of the triangle -/
  b : ℝ
  /-- The length of the third side of the triangle -/
  c : ℝ
  /-- The length of the first segment of side 'a' -/
  x : ℝ
  /-- The length of the second segment of side 'a' -/
  y : ℝ
  /-- The side 'a' is divided by the point of tangency -/
  side_division : a = x + y
  /-- All sides are positive -/
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  /-- All segments are positive -/
  pos_segments : 0 < x ∧ 0 < y

/-- Theorem about the sides of a triangle with an inscribed circle -/
theorem inscribed_triangle_sides (t : InscribedTriangle) 
  (h1 : t.r = 2)
  (h2 : t.x = 6)
  (h3 : t.y = 14) :
  (t.b = 7 ∧ t.c = 15) ∨ (t.b = 15 ∧ t.c = 7) := by
  sorry

end inscribed_triangle_sides_l519_51910


namespace faye_coloring_books_l519_51939

def coloring_books_problem (initial_books : ℝ) (first_giveaway : ℝ) (second_giveaway : ℝ) : Prop :=
  initial_books - first_giveaway - second_giveaway = 11.0

theorem faye_coloring_books :
  coloring_books_problem 48.0 34.0 3.0 := by
  sorry

end faye_coloring_books_l519_51939


namespace square_difference_divided_l519_51991

theorem square_difference_divided : (180^2 - 150^2) / 30 = 330 := by
  sorry

end square_difference_divided_l519_51991


namespace ella_reads_500_pages_l519_51989

/-- Represents the reading task for Ella and John -/
structure ReadingTask where
  total_pages : ℕ
  ella_pace : ℕ  -- seconds per page
  john_pace : ℕ  -- seconds per page

/-- Calculates the number of pages Ella should read -/
def pages_for_ella (task : ReadingTask) : ℕ :=
  (task.total_pages * task.john_pace) / (task.ella_pace + task.john_pace)

/-- Theorem stating that Ella should read 500 pages given the conditions -/
theorem ella_reads_500_pages (task : ReadingTask) 
  (h1 : task.total_pages = 900)
  (h2 : task.ella_pace = 40)
  (h3 : task.john_pace = 50) : 
  pages_for_ella task = 500 := by
  sorry

#eval pages_for_ella ⟨900, 40, 50⟩

end ella_reads_500_pages_l519_51989


namespace halfway_fraction_l519_51984

theorem halfway_fraction (a b c : ℚ) (ha : a = 1/4) (hb : b = 1/6) (hc : c = 1/3) :
  (a + b + c) / 3 = 1/4 := by
  sorry

end halfway_fraction_l519_51984


namespace square_area_with_perimeter_40_l519_51940

theorem square_area_with_perimeter_40 :
  ∀ s : ℝ, s > 0 → 4 * s = 40 → s * s = 100 := by
  sorry

end square_area_with_perimeter_40_l519_51940


namespace machine_work_time_l519_51905

theorem machine_work_time (x : ℝ) : x > 0 → 
  (1 / (x + 8) + 2 / (3 * x) + 1 / (2 * x) = 1 / (2 * x)) → 
  x = (-1 + Real.sqrt 97) / 6 :=
by sorry

end machine_work_time_l519_51905


namespace casper_window_problem_l519_51958

theorem casper_window_problem (total_windows : ℕ) (locked_windows : ℕ) : 
  total_windows = 8 → 
  locked_windows = 1 → 
  (total_windows - locked_windows) * (total_windows - locked_windows - 1) = 42 :=
by
  sorry

end casper_window_problem_l519_51958


namespace one_weighing_sufficient_l519_51987

/-- Represents a coin, which can be either real or counterfeit -/
inductive Coin
  | Real
  | Counterfeit

/-- Represents the result of weighing two coins -/
inductive WeighResult
  | Left  -- Left side is lighter
  | Right -- Right side is lighter
  | Equal -- Both sides are equal

/-- A function that simulates weighing two coins -/
def weigh (a b : Coin) : WeighResult :=
  match a, b with
  | Coin.Counterfeit, Coin.Real    => WeighResult.Left
  | Coin.Real, Coin.Counterfeit    => WeighResult.Right
  | Coin.Real, Coin.Real           => WeighResult.Equal
  | Coin.Counterfeit, Coin.Counterfeit => WeighResult.Equal

/-- A function that determines the counterfeit coin given three coins -/
def findCounterfeit (a b c : Coin) : Coin :=
  match weigh a b with
  | WeighResult.Left  => a
  | WeighResult.Right => b
  | WeighResult.Equal => c

theorem one_weighing_sufficient :
  ∀ (a b c : Coin),
  (∃! x, x = Coin.Counterfeit) →
  (a = Coin.Counterfeit ∨ b = Coin.Counterfeit ∨ c = Coin.Counterfeit) →
  findCounterfeit a b c = Coin.Counterfeit :=
by sorry

end one_weighing_sufficient_l519_51987


namespace solve_system_equations_solve_system_inequalities_l519_51961

-- Part 1: System of Equations
theorem solve_system_equations (x y : ℝ) :
  (3 * (y - 2) = x - 1) ∧ (2 * (x - 1) = 5 * y - 8) →
  x = 7 ∧ y = 4 := by sorry

-- Part 2: System of Inequalities
theorem solve_system_inequalities (x : ℝ) :
  (3 * x ≤ 2 * x + 3) ∧ ((x + 1) / 6 - 1 < (2 * x + 2) / 3) ↔
  -3 < x ∧ x ≤ 3 := by sorry

end solve_system_equations_solve_system_inequalities_l519_51961


namespace rose_price_is_seven_l519_51954

/-- Calculates the price per rose given the initial number of roses,
    remaining number of roses, and total earnings. -/
def price_per_rose (initial : ℕ) (remaining : ℕ) (earnings : ℕ) : ℚ :=
  earnings / (initial - remaining)

/-- Proves that the price per rose is 7 dollars given the problem conditions. -/
theorem rose_price_is_seven :
  price_per_rose 9 4 35 = 7 := by
  sorry

end rose_price_is_seven_l519_51954


namespace max_term_T_l519_51918

def geometric_sequence (a₁ : ℚ) (q : ℚ) : ℕ+ → ℚ :=
  fun n => a₁ * q ^ (n.val - 1)

def sum_geometric_sequence (a₁ : ℚ) (q : ℚ) : ℕ+ → ℚ :=
  fun n => a₁ * (1 - q^n.val) / (1 - q)

def T (S : ℕ+ → ℚ) : ℕ+ → ℚ :=
  fun n => S n + 1 / (S n)

theorem max_term_T 
  (a : ℕ+ → ℚ)
  (S : ℕ+ → ℚ)
  (h₁ : a 1 = 3/2)
  (h₂ : ∀ n, S n = sum_geometric_sequence (a 1) (-1/2) n)
  (h₃ : -2*(S 2) + 4*(S 4) = 2*(S 3))
  : ∀ n, T S n ≤ 13/6 ∧ T S 1 = 13/6 :=
sorry

end max_term_T_l519_51918


namespace competition_results_l519_51996

/-- Represents the weights of lifts for an athlete -/
structure AthleteLifts where
  first : ℕ
  second : ℕ

/-- The competition results -/
def Competition : Type :=
  AthleteLifts × AthleteLifts × AthleteLifts

def joe_total (c : Competition) : ℕ := c.1.first + c.1.second
def mike_total (c : Competition) : ℕ := c.2.1.first + c.2.1.second
def lisa_total (c : Competition) : ℕ := c.2.2.first + c.2.2.second

def joe_condition (c : Competition) : Prop :=
  2 * c.1.first = c.1.second + 300

def mike_condition (c : Competition) : Prop :=
  c.2.1.second = c.2.1.first + 200

def lisa_condition (c : Competition) : Prop :=
  c.2.2.first = 3 * c.2.2.second

theorem competition_results (c : Competition) 
  (h1 : joe_total c = 900)
  (h2 : mike_total c = 1100)
  (h3 : lisa_total c = 1000)
  (h4 : joe_condition c)
  (h5 : mike_condition c)
  (h6 : lisa_condition c) :
  c.1.first = 400 ∧ c.2.1.first = 450 ∧ c.2.2.second = 250 := by
  sorry

end competition_results_l519_51996


namespace distance_point_to_line_l519_51967

def vector_AB : Fin 3 → ℝ := ![1, 1, 2]
def vector_AC : Fin 3 → ℝ := ![2, 1, 1]

theorem distance_point_to_line :
  let distance := Real.sqrt (6 - (5 * Real.sqrt 6 / 6) ^ 2)
  distance = Real.sqrt 66 / 6 := by sorry

end distance_point_to_line_l519_51967


namespace fraction_equality_l519_51977

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : a / b = (2 * a) / (2 * b) := by
  sorry

end fraction_equality_l519_51977


namespace functional_equation_l519_51999

theorem functional_equation (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x - f y) = 1 - x - y) →
  (∀ x : ℝ, f x = 1/2 - x) := by
sorry

end functional_equation_l519_51999


namespace marble_problem_l519_51994

theorem marble_problem (a : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ)
  (h1 : angela = a)
  (h2 : brian = 2 * a)
  (h3 : caden = 3 * brian)
  (h4 : daryl = 6 * caden)
  (h5 : angela + brian + caden + daryl = 150) :
  a = 10 / 3 := by
sorry

end marble_problem_l519_51994


namespace opera_house_empty_seats_percentage_l519_51970

theorem opera_house_empty_seats_percentage
  (total_rows : ℕ)
  (seats_per_row : ℕ)
  (ticket_price : ℕ)
  (earnings : ℕ)
  (h1 : total_rows = 150)
  (h2 : seats_per_row = 10)
  (h3 : ticket_price = 10)
  (h4 : earnings = 12000) :
  (((total_rows * seats_per_row) - (earnings / ticket_price)) * 100) / (total_rows * seats_per_row) = 20 := by
  sorry

#check opera_house_empty_seats_percentage

end opera_house_empty_seats_percentage_l519_51970


namespace ken_cycling_distance_l519_51900

/-- Ken's cycling speed in miles per hour when it's raining -/
def rain_speed : ℝ := 30 * 3

/-- Ken's cycling speed in miles per hour when it's snowing -/
def snow_speed : ℝ := 10 * 3

/-- Number of rainy days in a week -/
def rainy_days : ℕ := 3

/-- Number of snowy days in a week -/
def snowy_days : ℕ := 4

/-- Hours Ken cycles per day -/
def hours_per_day : ℝ := 1

theorem ken_cycling_distance :
  rain_speed * rainy_days * hours_per_day + snow_speed * snowy_days * hours_per_day = 390 := by
  sorry

end ken_cycling_distance_l519_51900


namespace unique_solution_cube_equation_l519_51948

theorem unique_solution_cube_equation :
  ∀ x y z : ℤ, x^3 - 3*y^3 - 9*z^3 = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end unique_solution_cube_equation_l519_51948


namespace sqrt_3_irrational_l519_51913

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end sqrt_3_irrational_l519_51913


namespace inspection_result_l519_51927

/-- Given a set of products and a selection for inspection, 
    we define the total number of items and the sample size. -/
def inspection_setup (total_products : ℕ) (selected : ℕ) : 
  (ℕ × ℕ) :=
  (total_products, selected)

/-- Theorem stating that for 50 products with 10 selected,
    the total number of items is 50 and the sample size is 10. -/
theorem inspection_result : 
  inspection_setup 50 10 = (50, 10) := by
  sorry

end inspection_result_l519_51927


namespace books_read_total_l519_51902

theorem books_read_total (may june july : ℕ) 
  (h_may : may = 2) 
  (h_june : june = 6) 
  (h_july : july = 10) : 
  may + june + july = 18 := by
  sorry

end books_read_total_l519_51902


namespace min_value_of_3a_plus_1_l519_51920

theorem min_value_of_3a_plus_1 (a : ℝ) (h : 8 * a^2 + 6 * a + 5 = 2) :
  ∃ (min_val : ℝ), min_val = -5/4 ∧ ∀ (x : ℝ), 8 * x^2 + 6 * x + 5 = 2 → 3 * x + 1 ≥ min_val :=
by sorry

end min_value_of_3a_plus_1_l519_51920


namespace two_true_propositions_l519_51935

theorem two_true_propositions :
  let original := ∀ a : ℝ, a > -3 → a > 0
  let converse := ∀ a : ℝ, a > 0 → a > -3
  let inverse := ∀ a : ℝ, a ≤ -3 → a ≤ 0
  let contrapositive := ∀ a : ℝ, a ≤ 0 → a ≤ -3
  (¬original ∧ converse ∧ inverse ∧ ¬contrapositive) :=
by sorry

end two_true_propositions_l519_51935


namespace waddle_hop_difference_l519_51976

/-- The number of hops Winston takes between consecutive markers -/
def winston_hops : ℕ := 88

/-- The number of waddles Petra takes between consecutive markers -/
def petra_waddles : ℕ := 24

/-- The total number of markers -/
def total_markers : ℕ := 81

/-- The total distance in feet between the first and last marker -/
def total_distance : ℕ := 10560

/-- The length of Petra's waddle in feet -/
def petra_waddle_length : ℚ := total_distance / (petra_waddles * (total_markers - 1))

/-- The length of Winston's hop in feet -/
def winston_hop_length : ℚ := total_distance / (winston_hops * (total_markers - 1))

/-- The difference between Petra's waddle length and Winston's hop length -/
def length_difference : ℚ := petra_waddle_length - winston_hop_length

theorem waddle_hop_difference : length_difference = 4 := by
  sorry

end waddle_hop_difference_l519_51976


namespace quaternary_201_is_33_l519_51962

def quaternary_to_decimal (q : ℕ) : ℕ :=
  (q / 100) * 4^2 + ((q / 10) % 10) * 4^1 + (q % 10) * 4^0

theorem quaternary_201_is_33 : quaternary_to_decimal 201 = 33 := by
  sorry

end quaternary_201_is_33_l519_51962


namespace certain_number_proof_l519_51943

theorem certain_number_proof : ∃ x : ℚ, 346 * x = 173 * 240 ∧ x = 120 := by
  sorry

end certain_number_proof_l519_51943


namespace polynomial_sum_l519_51926

theorem polynomial_sum (x : ℝ) (h1 : x^5 - 3*x + 2 = 0) (h2 : x ≠ 1) :
  x^4 + x^3 + x^2 + x + 1 = 3 := by
  sorry

end polynomial_sum_l519_51926


namespace intersection_and_chord_length_l519_51908

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 2)^2 = 4

-- Define the line l₃
def line_l₃ (x y : ℝ) : Prop :=
  4*x - 3*y - 1 = 0

-- Theorem statement
theorem intersection_and_chord_length :
  ∃ (A B : ℝ × ℝ),
    A ≠ B ∧
    circle_C A.1 A.2 ∧
    circle_C B.1 B.2 ∧
    line_l₃ A.1 A.2 ∧
    line_l₃ B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 :=
by
  sorry

end intersection_and_chord_length_l519_51908


namespace ellipse_vertices_distance_l519_51981

/-- Given an ellipse with equation (x^2 / 45) + (y^2 / 11) = 1, 
    the distance between its vertices is 6√5 -/
theorem ellipse_vertices_distance : 
  ∀ (x y : ℝ), x^2/45 + y^2/11 = 1 → 
  ∃ (d : ℝ), d = 6 * Real.sqrt 5 ∧ d = 2 * Real.sqrt (max 45 11) :=
by sorry

end ellipse_vertices_distance_l519_51981


namespace inequality_solution_set_l519_51909

theorem inequality_solution_set (θ : ℝ) (x : ℝ) :
  (|x + Real.cos θ ^ 2| ≤ Real.sin θ ^ 2) ↔ (-1 ≤ x ∧ x ≤ -Real.cos (2 * θ)) :=
by sorry

end inequality_solution_set_l519_51909


namespace problem_statement_l519_51922

theorem problem_statement (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃! d : ℝ, d > 0 ∧ 1 / (a + d) + 1 / (b + d) + 1 / (c + d) = 2 / d ∧
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → a * x + b * y + c * z = x * y * z →
    x + y + z ≥ (2 / d) * Real.sqrt ((a + d) * (b + d) * (c + d)) :=
by sorry

end problem_statement_l519_51922


namespace quadratic_distinct_roots_l519_51964

/-- 
Theorem: For the quadratic equation x^2 - 6x + k = 0 to have two distinct real roots, k must be less than 9.
-/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x + k = 0 ∧ y^2 - 6*y + k = 0) → k < 9 :=
by sorry

end quadratic_distinct_roots_l519_51964


namespace min_modulus_m_for_real_roots_l519_51916

/-- Given a complex number m such that the equation x^2 + mx + 1 + 2i = 0 has real roots,
    the minimum value of |m| is sqrt(2 + 2sqrt(5)). -/
theorem min_modulus_m_for_real_roots (m : ℂ) : 
  (∃ x : ℝ, x^2 + m * x + (1 : ℂ) + 2*I = 0) → 
  ∀ m' : ℂ, (∃ x : ℝ, x^2 + m' * x + (1 : ℂ) + 2*I = 0) → Complex.abs m ≤ Complex.abs m' → 
  Complex.abs m = Real.sqrt (2 + 2 * Real.sqrt 5) :=
by sorry

end min_modulus_m_for_real_roots_l519_51916


namespace ninth_term_value_l519_51975

/-- A geometric sequence with a₁ = 2 and a₅ = 18 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ a 5 = 18 ∧ ∀ n m : ℕ, a (n + m) = a n * a m

theorem ninth_term_value (a : ℕ → ℝ) (h : geometric_sequence a) : a 9 = 162 := by
  sorry

end ninth_term_value_l519_51975


namespace union_of_A_and_B_l519_51959

-- Define the sets A and B
def A : Set Int := {-1, 1, 2, 4}
def B : Set Int := {-1, 0, 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2, 4} := by sorry

end union_of_A_and_B_l519_51959


namespace system_of_equations_solution_l519_51924

theorem system_of_equations_solution :
  ∃ (x y : ℚ), 3 * x - 4 * y = -7 ∧ 4 * x - 3 * y = 5 ∧ x = 41 / 7 ∧ y = 43 / 7 := by
  sorry

end system_of_equations_solution_l519_51924


namespace color_tv_price_l519_51963

theorem color_tv_price : ∃ (x : ℝ), x > 0 ∧ (1.4 * x * 0.8) - x = 144 ∧ x = 1200 := by
  sorry

end color_tv_price_l519_51963


namespace quadratic_discriminant_l519_51986

theorem quadratic_discriminant : 
  let a : ℝ := 1
  let b : ℝ := -4
  let c : ℝ := 3
  b^2 - 4*a*c = 4 := by sorry

end quadratic_discriminant_l519_51986


namespace sum_of_repeating_decimals_l519_51901

-- Define the repeating decimals
def repeating_2 : ℚ := 2 / 9
def repeating_03 : ℚ := 3 / 99
def repeating_0004 : ℚ := 4 / 9999
def repeating_00005 : ℚ := 5 / 99999

-- State the theorem
theorem sum_of_repeating_decimals :
  repeating_2 + repeating_03 + repeating_0004 + repeating_00005 = 56534 / 99999 := by
  sorry

end sum_of_repeating_decimals_l519_51901


namespace inequality_for_negative_reals_l519_51917

theorem inequality_for_negative_reals (a b : ℝ) : 
  a < b → b < 0 → a + 1/b < b + 1/a := by sorry

end inequality_for_negative_reals_l519_51917


namespace fencing_requirement_l519_51914

/-- Calculates the required fencing for a rectangular field -/
theorem fencing_requirement (area : ℝ) (uncovered_side : ℝ) : 
  area > 0 → uncovered_side > 0 → area = uncovered_side * (area / uncovered_side) →
  uncovered_side + 2 * (area / uncovered_side) = 25 := by
  sorry

#check fencing_requirement

end fencing_requirement_l519_51914


namespace common_tangent_sum_l519_51997

/-- Parabola P₁ -/
def P₁ (x y : ℚ) : Prop := y = x^2 + 52/25

/-- Parabola P₂ -/
def P₂ (x y : ℚ) : Prop := x = y^2 + 81/16

/-- Common tangent line L -/
def L (a b c : ℕ) (x y : ℚ) : Prop := a * x + b * y = c

/-- L has rational slope -/
def rational_slope (a b : ℕ) : Prop := ∃ (p q : ℤ), p ≠ 0 ∧ q ≠ 0 ∧ (a : ℚ) / b = p / q

theorem common_tangent_sum (a b c : ℕ) :
  (∀ x y : ℚ, P₁ x y → L a b c x y → (∃ t : ℚ, ∀ x' y', P₁ x' y' → L a b c x' y' → (x' - x)^2 + (y' - y)^2 ≤ t^2)) →
  (∀ x y : ℚ, P₂ x y → L a b c x y → (∃ t : ℚ, ∀ x' y', P₂ x' y' → L a b c x' y' → (x' - x)^2 + (y' - y)^2 ≤ t^2)) →
  rational_slope a b →
  a > 0 → b > 0 → c > 0 →
  Nat.gcd a (Nat.gcd b c) = 1 →
  a + b + c = 168 :=
sorry

end common_tangent_sum_l519_51997


namespace symmetry_wrt_origin_l519_51941

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem symmetry_wrt_origin :
  symmetric_point (4, -1) = (-4, 1) := by sorry

end symmetry_wrt_origin_l519_51941


namespace largest_number_in_l_pattern_l519_51966

/-- Represents the L-shaped pattern in the number arrangement --/
structure LPattern where
  largest : ℕ
  second : ℕ
  third : ℕ

/-- The sum of numbers in the L-shaped pattern is 2015 --/
def sum_is_2015 (p : LPattern) : Prop :=
  p.largest + p.second + p.third = 2015

/-- The L-shaped pattern follows the specific arrangement described --/
def valid_arrangement (p : LPattern) : Prop :=
  (p.second = p.largest - 6 ∧ p.third = p.largest - 7) ∨
  (p.second = p.largest - 7 ∧ p.third = p.largest - 8) ∨
  (p.second = p.largest - 1 ∧ p.third = p.largest - 7) ∨
  (p.second = p.largest - 1 ∧ p.third = p.largest - 8)

theorem largest_number_in_l_pattern :
  ∀ p : LPattern, sum_is_2015 p → valid_arrangement p → p.largest = 676 := by
  sorry

end largest_number_in_l_pattern_l519_51966


namespace prop_1_prop_2_prop_3_prop_4_false_l519_51965

-- Define the function f
def f (x b c : ℝ) : ℝ := x * abs x + b * x + c

-- Proposition ①
theorem prop_1 (x : ℝ) : f x 0 0 = -f (-x) 0 0 := by sorry

-- Proposition ②
theorem prop_2 : ∃! x : ℝ, f x 0 1 = 0 := by sorry

-- Proposition ③
theorem prop_3 (x b c : ℝ) : f x b c - c = -(f (-x) b c - c) := by sorry

-- Proposition ④ (false)
theorem prop_4_false : ∃ b c : ℝ, ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
  f x b c = 0 ∧ f y b c = 0 ∧ f z b c = 0 := by sorry

end prop_1_prop_2_prop_3_prop_4_false_l519_51965


namespace no_solution_composite_l519_51936

/-- Two polynomials P and Q that satisfy the given conditions -/
class SpecialPolynomials (P Q : ℝ → ℝ) : Prop where
  commutativity : ∀ x : ℝ, P (Q x) = Q (P x)
  no_solution : ∀ x : ℝ, P x ≠ Q x

/-- Theorem stating that if P and Q satisfy the special conditions,
    then P(P(x)) = Q(Q(x)) has no solutions -/
theorem no_solution_composite 
  (P Q : ℝ → ℝ) [SpecialPolynomials P Q] :
  ∀ x : ℝ, P (P x) ≠ Q (Q x) := by
  sorry

end no_solution_composite_l519_51936


namespace magic_square_solution_l519_51928

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a11 : ℚ
  a12 : ℚ
  a13 : ℚ
  a21 : ℚ
  a22 : ℚ
  a23 : ℚ
  a31 : ℚ
  a32 : ℚ
  a33 : ℚ
  sum_property : ∃ s : ℚ,
    a11 + a12 + a13 = s ∧
    a21 + a22 + a23 = s ∧
    a31 + a32 + a33 = s ∧
    a11 + a21 + a31 = s ∧
    a12 + a22 + a32 = s ∧
    a13 + a23 + a33 = s ∧
    a11 + a22 + a33 = s ∧
    a13 + a22 + a31 = s

/-- The theorem stating that y = 168.5 in the given magic square -/
theorem magic_square_solution :
  ∀ (ms : MagicSquare),
    ms.a11 = ms.a11 ∧  -- y (unknown)
    ms.a12 = 25 ∧
    ms.a13 = 81 ∧
    ms.a21 = 4 →
    ms.a11 = 168.5 := by
  sorry


end magic_square_solution_l519_51928


namespace complex_product_real_imag_parts_l519_51979

theorem complex_product_real_imag_parts : ∃ (m n : ℝ), 
  let Z : ℂ := (1 + Complex.I) * (2 + Complex.I^607)
  m = Z.re ∧ n = Z.im ∧ m * n = 3 := by
  sorry

end complex_product_real_imag_parts_l519_51979


namespace quadratic_equal_roots_l519_51938

/-- The roots of the quadratic equation ax² + 4ax + c = 0 are equal if and only if c = 4a, given that a ≠ 0 -/
theorem quadratic_equal_roots (a c : ℝ) (ha : a ≠ 0) :
  (∃ x : ℝ, ∀ y : ℝ, a * y^2 + 4 * a * y + c = 0 ↔ y = x) ↔ c = 4 * a :=
sorry

end quadratic_equal_roots_l519_51938


namespace max_n_minus_sum_digits_l519_51949

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The maximum value of n satisfying n - S(n) = 2007 is 2019 -/
theorem max_n_minus_sum_digits : 
  ∀ n : ℕ, n - sum_of_digits n = 2007 → n ≤ 2019 ∧ ∃ m : ℕ, m - sum_of_digits m = 2007 ∧ m = 2019 :=
sorry

end max_n_minus_sum_digits_l519_51949


namespace george_turning_25_l519_51992

/-- Represents George's age and bill exchange scenario --/
def GeorgeBirthdayProblem (n : ℕ) : Prop :=
  let billsReceived : ℕ := n
  let billsRemaining : ℚ := 0.8 * n
  let exchangeRate : ℚ := 1.5
  let totalExchange : ℚ := 12
  (exchangeRate * billsRemaining = totalExchange) ∧ (n + 15 = 25)

/-- Theorem stating that George is turning 25 years old --/
theorem george_turning_25 : ∃ n : ℕ, GeorgeBirthdayProblem n := by
  sorry

end george_turning_25_l519_51992


namespace divisibility_condition_l519_51904

theorem divisibility_condition (a b : ℕ+) :
  (a.val^2 + b.val^2 - a.val - b.val + 1) % (a.val * b.val) = 0 ↔ a = 1 ∧ b = 1 := by
  sorry

end divisibility_condition_l519_51904


namespace bookshelf_count_l519_51971

theorem bookshelf_count (books_per_shelf : ℕ) (total_books : ℕ) (shelf_count : ℕ) : 
  books_per_shelf = 15 → 
  total_books = 2250 → 
  shelf_count * books_per_shelf = total_books → 
  shelf_count = 150 := by
sorry

end bookshelf_count_l519_51971


namespace complex_sum_theorem_l519_51980

theorem complex_sum_theorem (a c d e f g : ℝ) : 
  let b : ℝ := 5
  let e : ℝ := -(a + c) + g
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) = g + 9 * Complex.I →
  d + f = 4 := by
sorry

end complex_sum_theorem_l519_51980
