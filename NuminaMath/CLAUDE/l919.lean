import Mathlib

namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l919_91924

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l919_91924


namespace NUMINAMATH_CALUDE_alien_head_volume_l919_91906

theorem alien_head_volume (surface_area : ℝ) (h : surface_area = 150) :
  let side_length := Real.sqrt (surface_area / 6)
  let volume := side_length ^ 3
  volume = 125 := by
  sorry

end NUMINAMATH_CALUDE_alien_head_volume_l919_91906


namespace NUMINAMATH_CALUDE_truncated_prism_cross_section_area_l919_91940

/-- Theorem: Square root of cross-section area in a truncated prism -/
theorem truncated_prism_cross_section_area 
  (S' S Q : ℝ) (n m : ℕ) 
  (h1 : 0 < S') (h2 : 0 < S) (h3 : 0 < Q) (h4 : 0 < n) (h5 : 0 < m) :
  Real.sqrt Q = (n * Real.sqrt S + m * Real.sqrt S') / (n + m) := by
  sorry

end NUMINAMATH_CALUDE_truncated_prism_cross_section_area_l919_91940


namespace NUMINAMATH_CALUDE_james_fish_catch_l919_91914

/-- The amount of trout James caught in pounds -/
def trout : ℝ := 200

/-- The amount of salmon James caught in pounds -/
def salmon : ℝ := 1.5 * trout

/-- The amount of tuna James caught in pounds -/
def tuna : ℝ := 2 * trout

/-- The total amount of fish James caught in pounds -/
def total_fish : ℝ := trout + salmon + tuna

theorem james_fish_catch : total_fish = 900 := by
  sorry

end NUMINAMATH_CALUDE_james_fish_catch_l919_91914


namespace NUMINAMATH_CALUDE_two_players_goals_l919_91913

theorem two_players_goals (total_goals : ℕ) (player1_goals player2_goals : ℕ) : 
  total_goals = 300 →
  player1_goals = player2_goals →
  player1_goals + player2_goals = total_goals / 5 →
  player1_goals = 30 ∧ player2_goals = 30 := by
  sorry

end NUMINAMATH_CALUDE_two_players_goals_l919_91913


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l919_91912

theorem arithmetic_sequence_length :
  ∀ (a₁ l d : ℤ),
  a₁ = -48 →
  d = 6 →
  l = 78 →
  ∃ n : ℕ,
    n > 0 ∧
    l = a₁ + d * (n - 1) ∧
    n = 22 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l919_91912


namespace NUMINAMATH_CALUDE_complex_expression_equality_l919_91948

theorem complex_expression_equality : 
  (125 : ℝ) ^ (1/3) - (-Real.sqrt 3)^2 + (1 + 1/Real.sqrt 2 - Real.sqrt 2) * Real.sqrt 2 - (-1)^2023 = Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l919_91948


namespace NUMINAMATH_CALUDE_power_of_product_squared_l919_91907

theorem power_of_product_squared (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_squared_l919_91907


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l919_91917

/-- A cubic equation with parameter p has three natural number roots -/
def has_three_natural_roots (p : ℝ) : Prop :=
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (5 * (x : ℝ)^3 - 5*(p+1)*(x : ℝ)^2 + (71*p-1)*(x : ℝ) + 1 = 66*p) ∧
    (5 * (y : ℝ)^3 - 5*(p+1)*(y : ℝ)^2 + (71*p-1)*(y : ℝ) + 1 = 66*p) ∧
    (5 * (z : ℝ)^3 - 5*(p+1)*(z : ℝ)^2 + (71*p-1)*(z : ℝ) + 1 = 66*p)

/-- If a cubic equation with parameter p has three natural number roots, then p = 76 -/
theorem cubic_equation_roots (p : ℝ) :
  has_three_natural_roots p → p = 76 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l919_91917


namespace NUMINAMATH_CALUDE_tissue_cost_theorem_l919_91908

/-- Calculates the total cost of tissues given the number of boxes, packs per box,
    tissues per pack, and price per tissue. -/
def totalCost (boxes : ℕ) (packsPerBox : ℕ) (tissuesPerPack : ℕ) (pricePerTissue : ℚ) : ℚ :=
  boxes * packsPerBox * tissuesPerPack * pricePerTissue

/-- Proves that the total cost of 10 boxes of tissues is $1,000 given the specified conditions. -/
theorem tissue_cost_theorem :
  totalCost 10 20 100 (5 / 100) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_tissue_cost_theorem_l919_91908


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l919_91950

theorem prime_sum_theorem (p q r s : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧
  p < q ∧ q < r ∧ r < s ∧
  1 - 1/p - 1/q - 1/r - 1/s = 1/(p*q*r*s) →
  p + q + r + s = 55 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l919_91950


namespace NUMINAMATH_CALUDE_probability_all_even_sum_l919_91982

/-- The number of tiles -/
def num_tiles : ℕ := 10

/-- The number of players -/
def num_players : ℕ := 3

/-- The number of tiles each player selects -/
def tiles_per_player : ℕ := 3

/-- The set of tile numbers -/
def tile_set : Finset ℕ := Finset.range num_tiles

/-- A function that returns true if a sum is even -/
def is_even_sum (sum : ℕ) : Prop := sum % 2 = 0

/-- The probability of a single player getting an even sum -/
def prob_even_sum_single : ℚ := 70 / 120

theorem probability_all_even_sum :
  (prob_even_sum_single ^ num_players : ℚ) = 343 / 1728 := by sorry

end NUMINAMATH_CALUDE_probability_all_even_sum_l919_91982


namespace NUMINAMATH_CALUDE_triangle_translation_l919_91927

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a triangle using three points
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

-- Define a translation function
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

-- Define the theorem
theorem triangle_translation :
  let ABC := Triangle.mk
    (Point.mk (-5) 0)
    (Point.mk 4 0)
    (Point.mk 2 5)
  let E := translate ABC.A 2 (-1)
  let F := translate ABC.B 2 (-1)
  let G := translate ABC.C 2 (-1)
  let EFG := Triangle.mk E F G
  E = Point.mk (-3) (-1) ∧
  F = Point.mk 6 (-1) ∧
  G = Point.mk 4 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_translation_l919_91927


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l919_91959

theorem rectangle_length_proof (b : ℝ) (h1 : b > 0) : 
  (2 * b - 5) * (b + 5) = 2 * b^2 + 75 → 2 * b = 40 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_proof_l919_91959


namespace NUMINAMATH_CALUDE_katies_cupcakes_l919_91902

/-- Proves that Katie initially made 26 cupcakes given the conditions of the problem -/
theorem katies_cupcakes (X : ℕ) : X = 26 :=
  by
    -- Define the conditions
    have sold : ℕ := 20
    have made_more : ℕ := 20
    have current : ℕ := 26

    -- State the relationship between initial, sold, made more, and current cupcakes
    have h : X - sold + made_more = current := by sorry

    -- Prove that X = 26
    sorry

end NUMINAMATH_CALUDE_katies_cupcakes_l919_91902


namespace NUMINAMATH_CALUDE_petya_more_likely_to_win_l919_91962

/-- Represents the game setup with two boxes of candies --/
structure CandyGame where
  total_candies : ℕ
  prob_two_caramels : ℝ

/-- Defines the game rules --/
def game : CandyGame :=
  { total_candies := 25
  , prob_two_caramels := 0.54 }

/-- Calculates the probability of Vasya winning --/
def vasya_win_prob (g : CandyGame) : ℝ :=
  1 - g.prob_two_caramels

/-- Calculates the probability of Petya winning --/
def petya_win_prob (g : CandyGame) : ℝ :=
  1 - vasya_win_prob g

/-- Theorem stating that Petya has a higher chance of winning --/
theorem petya_more_likely_to_win :
  petya_win_prob game > vasya_win_prob game :=
sorry

end NUMINAMATH_CALUDE_petya_more_likely_to_win_l919_91962


namespace NUMINAMATH_CALUDE_rectangle_area_l919_91961

theorem rectangle_area (square_area : ℝ) (rectangle_length_factor : ℝ) : 
  square_area = 64 →
  rectangle_length_factor = 3 →
  let square_side := Real.sqrt square_area
  let rectangle_width := square_side
  let rectangle_length := rectangle_length_factor * rectangle_width
  rectangle_width * rectangle_length = 192 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l919_91961


namespace NUMINAMATH_CALUDE_gcd_1337_382_l919_91942

theorem gcd_1337_382 : Nat.gcd 1337 382 = 191 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1337_382_l919_91942


namespace NUMINAMATH_CALUDE_rice_distribution_l919_91966

theorem rice_distribution (total_rice : ℝ) (difference : ℝ) (fraction : ℝ) : 
  total_rice = 50 →
  difference = 20 →
  fraction * total_rice = (1 - fraction) * total_rice + difference →
  fraction = 7/10 := by
sorry

end NUMINAMATH_CALUDE_rice_distribution_l919_91966


namespace NUMINAMATH_CALUDE_triangle_ratio_l919_91901

/-- In a triangle ABC, given that a * sin(A) * sin(B) + b * cos²(A) = √3 * a, 
    prove that b/a = √3 -/
theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) : 
  (a > 0) → 
  (b > 0) → 
  (c > 0) → 
  (A > 0) → (A < π) →
  (B > 0) → (B < π) →
  (C > 0) → (C < π) →
  (A + B + C = π) →
  (a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = Real.sqrt 3 * a) →
  (b / a = Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l919_91901


namespace NUMINAMATH_CALUDE_increasing_and_second_derivative_l919_91933

open Set

-- Define the properties
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

def HasPositiveSecondDerivative (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a < x ∧ x < b → (deriv (deriv f)) x > 0

-- Theorem statement
theorem increasing_and_second_derivative (f : ℝ → ℝ) (a b : ℝ) :
  (HasPositiveSecondDerivative f a b → IsIncreasing f a b) ∧
  ∃ g : ℝ → ℝ, IsIncreasing g a b ∧ ∃ x, a < x ∧ x < b ∧ (deriv (deriv g)) x ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_increasing_and_second_derivative_l919_91933


namespace NUMINAMATH_CALUDE_percent_commutation_l919_91935

theorem percent_commutation (x : ℝ) (h : 0.3 * (0.4 * x) = 36) :
  0.4 * (0.3 * x) = 0.3 * (0.4 * x) := by
  sorry

end NUMINAMATH_CALUDE_percent_commutation_l919_91935


namespace NUMINAMATH_CALUDE_price_increase_percentage_l919_91973

theorem price_increase_percentage (old_price new_price : ℝ) 
  (h1 : old_price = 300)
  (h2 : new_price = 330) :
  ((new_price - old_price) / old_price) * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l919_91973


namespace NUMINAMATH_CALUDE_min_fraction_sum_l919_91903

theorem min_fraction_sum (c d : ℕ) (hc : c > 0) (hd : d > 0) (hcd : c > d) 
  (hodd : Odd (c + d)) :
  (c + d : ℚ) / (c - d) + (c - d : ℚ) / (c + d) ≥ 10 / 3 ∧
  ∃ (c' d' : ℕ), c' > 0 ∧ d' > 0 ∧ c' > d' ∧ Odd (c' + d') ∧
    (c' + d' : ℚ) / (c' - d') + (c' - d' : ℚ) / (c' + d') = 10 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l919_91903


namespace NUMINAMATH_CALUDE_can_open_lock_can_toggle_single_switch_l919_91996

/-- Represents the state of a switch (on or off) -/
inductive SwitchState
| On
| Off

/-- Represents the position of a switch on the 4x4 board -/
structure Position where
  row : Fin 4
  col : Fin 4

/-- Represents the state of the entire 4x4 digital lock -/
def LockState := Position → SwitchState

/-- Toggles a switch state -/
def toggleSwitch (s : SwitchState) : SwitchState :=
  match s with
  | SwitchState.On => SwitchState.Off
  | SwitchState.Off => SwitchState.On

/-- Applies a move to the lock state -/
def applyMove (state : LockState) (pos : Position) : LockState :=
  fun p => if p.row = pos.row || p.col = pos.col then toggleSwitch (state p) else state p

/-- Checks if all switches in the lock state are on -/
def allSwitchesOn (state : LockState) : Prop :=
  ∀ p : Position, state p = SwitchState.On

/-- Theorem: It is always possible to open the lock from any initial configuration -/
theorem can_open_lock (initialState : LockState) :
  ∃ (moves : List Position), allSwitchesOn (moves.foldl applyMove initialState) := by sorry

/-- Theorem: It is possible to toggle only one switch through a sequence of moves -/
theorem can_toggle_single_switch (initialState : LockState) (targetPos : Position) :
  ∃ (moves : List Position),
    let finalState := moves.foldl applyMove initialState
    (∀ p : Position, p ≠ targetPos → finalState p = initialState p) ∧
    finalState targetPos ≠ initialState targetPos := by sorry

end NUMINAMATH_CALUDE_can_open_lock_can_toggle_single_switch_l919_91996


namespace NUMINAMATH_CALUDE_equation_solution_expression_simplification_l919_91993

-- Part 1: Equation solution
theorem equation_solution (x : ℝ) :
  (x + 3) / (x - 3) - 4 / (x + 3) = 1 ↔ x = -15 :=
sorry

-- Part 2: Expression simplification
theorem expression_simplification (x : ℝ) (h : x ≠ 2) (h' : x ≠ -3) :
  (x - 3) / (x - 2) / (x + 2 - 5 / (x - 2)) = 1 / (x + 3) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_expression_simplification_l919_91993


namespace NUMINAMATH_CALUDE_increasing_root_m_value_l919_91951

theorem increasing_root_m_value (m : ℝ) : 
  (∃ x : ℝ, (2 * x + 1) / (x - 3) = m / (3 - x) + 1 ∧ 
   ∀ y : ℝ, y > x → (2 * y + 1) / (y - 3) > m / (3 - y) + 1) → 
  m = -7 := by
sorry

end NUMINAMATH_CALUDE_increasing_root_m_value_l919_91951


namespace NUMINAMATH_CALUDE_optimal_difference_optimal_difference_four_stars_l919_91992

/-- Represents the state of the game board -/
structure GameBoard where
  n : ℕ
  minuend : List ℕ
  subtrahend : List ℕ

/-- Represents a player in the game -/
inductive Player
  | First
  | Second

/-- Represents a move in the game -/
structure Move where
  digit : ℕ
  position : ℕ
  player : Player

/-- The game state after a sequence of moves -/
def gameState (initial : GameBoard) (moves : List Move) : GameBoard :=
  sorry

/-- The difference between minuend and subtrahend on the game board -/
def boardDifference (board : GameBoard) : ℕ :=
  sorry

/-- Optimal strategy for the first player -/
def firstPlayerStrategy (board : GameBoard) : Move :=
  sorry

/-- Optimal strategy for the second player -/
def secondPlayerStrategy (board : GameBoard) (digit : ℕ) : Move :=
  sorry

/-- The main theorem stating the optimal difference -/
theorem optimal_difference (n : ℕ) :
  ∀ (moves : List Move),
    boardDifference (gameState (GameBoard.mk n [] []) moves) ≤ 4 * 10^(n-1) ∧
    boardDifference (gameState (GameBoard.mk n [] []) moves) ≥ 4 * 10^(n-1) :=
  sorry

/-- Corollary for the specific case of n = 4 -/
theorem optimal_difference_four_stars :
  ∀ (moves : List Move),
    boardDifference (gameState (GameBoard.mk 4 [] []) moves) = 4000 :=
  sorry

end NUMINAMATH_CALUDE_optimal_difference_optimal_difference_four_stars_l919_91992


namespace NUMINAMATH_CALUDE_distance_origin_to_point_l919_91916

/-- The distance between the origin (0,0) and the point (1, √3) in a Cartesian coordinate system is 2. -/
theorem distance_origin_to_point :
  let A : ℝ × ℝ := (1, Real.sqrt 3)
  Real.sqrt ((A.1 - 0)^2 + (A.2 - 0)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_origin_to_point_l919_91916


namespace NUMINAMATH_CALUDE_vector_magnitude_l919_91980

def a : Fin 2 → ℝ := ![(-2 : ℝ), 1]
def b (k : ℝ) : Fin 2 → ℝ := ![k, -3]
def c : Fin 2 → ℝ := ![1, 2]

theorem vector_magnitude (k : ℝ) :
  (∀ i : Fin 2, (a i - 2 * b k i) * c i = 0) →
  Real.sqrt ((b k 0)^2 + (b k 1)^2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l919_91980


namespace NUMINAMATH_CALUDE_triangle_problem_l919_91939

open Real

theorem triangle_problem (A B C a b c : Real) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  (cos C) / (cos B) = (2 * a - c) / b →
  tan (A + π/4) = 7 →
  B = π/3 ∧ cos C = (3 * sqrt 3 - 4) / 10 := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l919_91939


namespace NUMINAMATH_CALUDE_remainder_1732_base12_div_9_l919_91990

/-- Converts a base-12 number to base-10 --/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (12 ^ i)) 0

/-- The base-12 representation of 1732₁₂ --/
def number_1732_base12 : List Nat := [2, 3, 7, 1]

theorem remainder_1732_base12_div_9 :
  (base12ToBase10 number_1732_base12) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1732_base12_div_9_l919_91990


namespace NUMINAMATH_CALUDE_new_person_weight_l919_91998

/-- Given a group of 8 people where one person weighing 65 kg is replaced by a new person,
    and the average weight of the group increases by 4.2 kg,
    prove that the weight of the new person is 98.6 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_increase : Real) (replaced_weight : Real) :
  initial_count = 8 →
  weight_increase = 4.2 →
  replaced_weight = 65 →
  (initial_count : Real) * weight_increase + replaced_weight = 98.6 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l919_91998


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l919_91952

theorem difference_of_squares_special_case : (1025 : ℤ) * 1025 - 1023 * 1027 = 4 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l919_91952


namespace NUMINAMATH_CALUDE_range_of_a_l919_91970

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x + 4 / x - 1 - a^2 + 2*a > 0) → 
  -1 < a ∧ a < 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l919_91970


namespace NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l919_91900

/-- Given an arithmetic sequence where the 4th term is 23 and the 6th term is 41, the 8th term is 59. -/
theorem arithmetic_sequence_8th_term 
  (a : ℝ) (d : ℝ) -- first term and common difference
  (h1 : a + 3*d = 23) -- 4th term condition
  (h2 : a + 5*d = 41) -- 6th term condition
  : a + 7*d = 59 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l919_91900


namespace NUMINAMATH_CALUDE_shoebox_plausibility_l919_91969

/-- Represents a rectangular prism object -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents possible objects that could match the given dimensions -/
inductive PossibleObject
  | PencilCase
  | MathTextbook
  | Bookshelf
  | Shoebox

/-- Determines if the given dimensions are plausible for a shoebox -/
def is_plausible_shoebox (prism : RectangularPrism) : Prop :=
  prism.length = 35 ∧ prism.width = 20 ∧ prism.height = 15

/-- Theorem stating that a rectangular prism with given dimensions could be a shoebox -/
theorem shoebox_plausibility (prism : RectangularPrism) 
  (h : is_plausible_shoebox prism) : 
  ∃ obj : PossibleObject, obj = PossibleObject.Shoebox := by
  sorry

end NUMINAMATH_CALUDE_shoebox_plausibility_l919_91969


namespace NUMINAMATH_CALUDE_exponent_multiplication_l919_91928

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l919_91928


namespace NUMINAMATH_CALUDE_negation_of_implication_l919_91968

theorem negation_of_implication (x : ℝ) :
  ¬(x > 1 → x > 0) ↔ (x ≤ 1 → x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l919_91968


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l919_91937

def P (x : ℝ) : ℝ := x^3 - 7*x^2 + 14*x - 8

theorem roots_of_polynomial :
  (∀ x : ℝ, P x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 4) ∧
  (∀ x : ℝ, (x - 1) * (x - 2) * (x - 4) = P x) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l919_91937


namespace NUMINAMATH_CALUDE_quadratic_sum_l919_91985

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℚ) : ℚ → ℚ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℚ) :
  (∃ f : ℚ → ℚ, f = quadratic a b c ∧
    (∀ x, f x = a * x^2 + b * x + c) ∧
    (f 3 = -2) ∧
    (∀ x, f (6 - x) = f x) ∧
    (f 0 = 5)) →
  a + b + c = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l919_91985


namespace NUMINAMATH_CALUDE_min_nSn_is_neg_nine_l919_91915

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_correct : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The theorem statement -/
theorem min_nSn_is_neg_nine (seq : ArithmeticSequence) (m : ℕ) (h_m : m ≥ 2) 
    (h_Sm_minus_one : seq.S (m - 1) = -2)
    (h_Sm : seq.S m = 0)
    (h_Sm_plus_one : seq.S (m + 1) = 3) :
    (∃ n : ℕ, seq.S n * n = -9) ∧ (∀ n : ℕ, seq.S n * n ≥ -9) := by
  sorry

end NUMINAMATH_CALUDE_min_nSn_is_neg_nine_l919_91915


namespace NUMINAMATH_CALUDE_baseball_card_pages_l919_91949

def number_of_pages (packs : ℕ) (cards_per_pack : ℕ) (cards_per_page : ℕ) : ℕ :=
  (packs * cards_per_pack) / cards_per_page

theorem baseball_card_pages : number_of_pages 60 7 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_pages_l919_91949


namespace NUMINAMATH_CALUDE_three_digit_remainder_problem_l919_91943

theorem three_digit_remainder_problem :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, 100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 3 ∧ n % 8 = 6 ∧ n % 12 = 8) ∧
    (∀ n, 100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 3 ∧ n % 8 = 6 ∧ n % 12 = 8 → n ∈ s) ∧
    s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_remainder_problem_l919_91943


namespace NUMINAMATH_CALUDE_stating_grouping_count_is_762_l919_91988

/-- Represents the number of tour guides -/
def num_guides : ℕ := 3

/-- Represents the number of tourists -/
def num_tourists : ℕ := 8

/-- 
Calculates the number of ways to distribute tourists among guides
where one guide has no tourists and the other two have at least one tourist each.
-/
def grouping_count : ℕ := sorry

/-- 
Theorem stating that the number of possible groupings
under the given conditions is 762.
-/
theorem grouping_count_is_762 : grouping_count = 762 := by sorry

end NUMINAMATH_CALUDE_stating_grouping_count_is_762_l919_91988


namespace NUMINAMATH_CALUDE_M_factorization_l919_91983

/-- The polynomial M(x, y, z) = x^3 + y^3 + z^3 - 3xyz -/
def M (x y z : ℝ) : ℝ := x^3 + y^3 + z^3 - 3*x*y*z

/-- The factorization of M(x, y, z) -/
theorem M_factorization (x y z : ℝ) :
  M x y z = (x + y + z) * (x^2 + y^2 + z^2 - x*y - y*z - z*x) := by
  sorry

end NUMINAMATH_CALUDE_M_factorization_l919_91983


namespace NUMINAMATH_CALUDE_shelter_dogs_l919_91984

theorem shelter_dogs (total_animals cats : ℕ) 
  (h1 : total_animals = 1212)
  (h2 : cats = 645) : 
  total_animals - cats = 567 := by
sorry

end NUMINAMATH_CALUDE_shelter_dogs_l919_91984


namespace NUMINAMATH_CALUDE_upper_limit_proof_l919_91945

theorem upper_limit_proof (x : ℝ) (upper_limit : ℝ) : 
  (3 < x ∧ x < 8) → (6 < x ∧ x < upper_limit) → x = 7 → upper_limit > 7 := by
sorry

end NUMINAMATH_CALUDE_upper_limit_proof_l919_91945


namespace NUMINAMATH_CALUDE_grocery_spending_fraction_l919_91987

theorem grocery_spending_fraction (initial_amount : ℝ) (magazine_fraction : ℝ) (final_amount : ℝ) 
  (h1 : initial_amount = 600)
  (h2 : magazine_fraction = 1/4)
  (h3 : final_amount = 360) :
  ∃ F : ℝ, 
    0 ≤ F ∧ F ≤ 1 ∧
    final_amount = (1 - F) * initial_amount * (1 - magazine_fraction) ∧
    F = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_grocery_spending_fraction_l919_91987


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l919_91967

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^2 ≥ 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l919_91967


namespace NUMINAMATH_CALUDE_triangle_angle_bisectors_l919_91981

/-- Given a triangle ABC with sides a, b, and c, this theorem proves the formulas for the lengths of its angle bisectors. -/
theorem triangle_angle_bisectors 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (a₁ b₁ cc₁ : ℝ),
    a₁ = a * c / (a + b) ∧
    b₁ = b * c / (a + b) ∧
    cc₁^2 = a * b * (1 - c^2 / (a + b)^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_bisectors_l919_91981


namespace NUMINAMATH_CALUDE_original_fraction_l919_91991

theorem original_fraction (x y : ℚ) : 
  x / (y + 1) = 1 / 2 → (x + 1) / y = 1 → x / y = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_l919_91991


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l919_91971

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n / d : ℚ) = 34 / 99 ∧ 
  (∀ (a b : ℕ), (a / b : ℚ) = 34 / 99 → b ≤ d) ∧ 
  n + d = 133 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l919_91971


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_power_2011_l919_91974

theorem last_four_digits_of_5_power_2011 : ∃ n : ℕ, 5^2011 ≡ 8125 [ZMOD 10000] :=
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_power_2011_l919_91974


namespace NUMINAMATH_CALUDE_solution_set_f_less_g_min_a_for_inequality_l919_91911

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2| - 3
def g (x : ℝ) : ℝ := |x + 3|

-- Theorem for the first part of the problem
theorem solution_set_f_less_g :
  {x : ℝ | f x < g x} = {x : ℝ | x > -2} :=
sorry

-- Theorem for the second part of the problem
theorem min_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, f x < g x + a) ↔ a > 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_g_min_a_for_inequality_l919_91911


namespace NUMINAMATH_CALUDE_reduced_oil_price_l919_91932

/-- Represents the price reduction scenario for oil --/
structure OilPriceReduction where
  original_price : ℝ
  reduced_price : ℝ
  reduction_percentage : ℝ
  additional_quantity : ℝ
  total_cost : ℝ

/-- Theorem stating the reduced price of oil given the conditions --/
theorem reduced_oil_price 
  (scenario : OilPriceReduction)
  (h1 : scenario.reduction_percentage = 0.4)
  (h2 : scenario.additional_quantity = 8)
  (h3 : scenario.total_cost = 2400)
  (h4 : scenario.reduced_price = scenario.original_price * (1 - scenario.reduction_percentage))
  (h5 : scenario.total_cost = (scenario.total_cost / scenario.original_price + scenario.additional_quantity) * scenario.reduced_price) :
  scenario.reduced_price = 120 := by
  sorry

end NUMINAMATH_CALUDE_reduced_oil_price_l919_91932


namespace NUMINAMATH_CALUDE_complex_absolute_value_sum_l919_91930

theorem complex_absolute_value_sum : 
  Complex.abs (3 - 5*I) + Complex.abs (3 + 5*I) + Complex.abs (1 + 5*I) = 2 * Real.sqrt 34 + Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_sum_l919_91930


namespace NUMINAMATH_CALUDE_problem_solution_l919_91904

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  (a + b > 1/2) ∧ 
  (a + b < 1) ∧ 
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 1 → 2/x + 1/y ≥ 8) ∧
  (a * b ≤ 1/8) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l919_91904


namespace NUMINAMATH_CALUDE_complement_of_B_l919_91976

def A : Set ℕ := {1, 2, 3}
def B (a : ℕ) : Set ℕ := {a + 2, a}

theorem complement_of_B (a : ℕ) (h : A ∩ B a = B a) : 
  (A \ B a) = {2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_B_l919_91976


namespace NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l919_91975

theorem joel_age_when_dad_twice_as_old (joel_current_age : ℕ) (dad_current_age : ℕ) : 
  joel_current_age = 5 → dad_current_age = 32 →
  ∃ (years : ℕ), dad_current_age + years = 2 * (joel_current_age + years) ∧ joel_current_age + years = 27 :=
by sorry

end NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l919_91975


namespace NUMINAMATH_CALUDE_augmented_matrix_sum_l919_91918

/-- Given an augmented matrix representing a system of linear equations and its solution,
    prove that the sum of certain elements in the matrix equals 10. -/
theorem augmented_matrix_sum (m n : ℝ) : 
  (∃ (A : Matrix (Fin 2) (Fin 3) ℝ), 
    A = ![![m, 0, 6],
         ![0, 3, n]] ∧ 
    (∀ (x y : ℝ), x = -3 ∧ y = 4 → m * x = 6 ∧ 3 * y = n)) →
  m + n = 10 := by
  sorry

end NUMINAMATH_CALUDE_augmented_matrix_sum_l919_91918


namespace NUMINAMATH_CALUDE_university_tuition_cost_l919_91986

def cost_first_8_years : ℕ := 8 * 10000
def cost_next_10_years : ℕ := 10 * 20000
def total_raising_cost : ℕ := cost_first_8_years + cost_next_10_years
def johns_contribution : ℕ := total_raising_cost / 2
def total_cost_with_tuition : ℕ := 265000

theorem university_tuition_cost :
  total_cost_with_tuition - johns_contribution = 125000 :=
by sorry

end NUMINAMATH_CALUDE_university_tuition_cost_l919_91986


namespace NUMINAMATH_CALUDE_readers_all_genres_l919_91999

theorem readers_all_genres (total : ℕ) (sci_fi : ℕ) (literary : ℕ) (non_fiction : ℕ)
  (sci_fi_literary : ℕ) (sci_fi_non_fiction : ℕ) (literary_non_fiction : ℕ) :
  total = 500 →
  sci_fi = 320 →
  literary = 200 →
  non_fiction = 150 →
  sci_fi_literary = 120 →
  sci_fi_non_fiction = 80 →
  literary_non_fiction = 60 →
  ∃ (all_genres : ℕ),
    all_genres = 90 ∧
    total = sci_fi + literary + non_fiction -
      (sci_fi_literary + sci_fi_non_fiction + literary_non_fiction) + all_genres :=
by
  sorry

end NUMINAMATH_CALUDE_readers_all_genres_l919_91999


namespace NUMINAMATH_CALUDE_g_diverges_from_negative_two_l919_91963

def g (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem g_diverges_from_negative_two :
  ∀ n : ℕ, ∃ M : ℝ, M > 0 ∧ 
  (n.iterate g (-2) > M ∧ n.iterate g (-2) > n.pred.iterate g (-2)) :=
sorry

end NUMINAMATH_CALUDE_g_diverges_from_negative_two_l919_91963


namespace NUMINAMATH_CALUDE_power_three_plus_four_mod_five_l919_91997

theorem power_three_plus_four_mod_five : 3^101 + 4 ≡ 2 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_power_three_plus_four_mod_five_l919_91997


namespace NUMINAMATH_CALUDE_prob_three_green_apples_l919_91994

/-- The probability of picking 3 green apples out of 10 apples, where 4 are green -/
theorem prob_three_green_apples (total : ℕ) (green : ℕ) (pick : ℕ)
  (h1 : total = 10) (h2 : green = 4) (h3 : pick = 3) :
  (Nat.choose green pick : ℚ) / (Nat.choose total pick) = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_green_apples_l919_91994


namespace NUMINAMATH_CALUDE_candy_mixture_cost_l919_91947

/-- Proves the cost of the second candy in a mixture given specific conditions --/
theorem candy_mixture_cost 
  (weight_first : ℝ) 
  (cost_first : ℝ) 
  (weight_total : ℝ) 
  (cost_mixture : ℝ) : 
  weight_first = 25 ∧ 
  cost_first = 8 ∧ 
  weight_total = 75 ∧ 
  cost_mixture = 6 → 
  (cost_mixture * weight_total - cost_first * weight_first) / (weight_total - weight_first) = 5 := by
sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_l919_91947


namespace NUMINAMATH_CALUDE_train_length_l919_91934

/-- The length of a train that crosses a platform of equal length in one minute at 126 km/hr -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (train_length platform_length : ℝ) :
  train_speed = 126 * 1000 / 3600 →
  crossing_time = 60 →
  train_length = platform_length →
  train_length * 2 = train_speed * crossing_time →
  train_length = 1050 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l919_91934


namespace NUMINAMATH_CALUDE_intersection_line_canonical_equations_l919_91923

/-- The canonical equations of the intersection line of two planes -/
theorem intersection_line_canonical_equations 
  (plane1 : ℝ → ℝ → ℝ → ℝ) 
  (plane2 : ℝ → ℝ → ℝ → ℝ) 
  (h1 : ∀ x y z, plane1 x y z = 3*x + 4*y - 2*z + 1)
  (h2 : ∀ x y z, plane2 x y z = 2*x - 4*y + 3*z + 4) :
  ∃ (t : ℝ), ∀ x y z, 
    plane1 x y z = 0 ∧ plane2 x y z = 0 ↔ 
    (x + 1) / 4 = (y - 1/2) / (-13) ∧ (y - 1/2) / (-13) = z / (-20) ∧ 
    x = -1 + 4*t ∧ y = 1/2 - 13*t ∧ z = -20*t :=
sorry

end NUMINAMATH_CALUDE_intersection_line_canonical_equations_l919_91923


namespace NUMINAMATH_CALUDE_stock_price_calculation_stock_price_problem_l919_91972

theorem stock_price_calculation (initial_price : ℝ) 
  (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  let final_price := price_after_first_year * (1 - second_year_decrease)
  final_price

theorem stock_price_problem : 
  stock_price_calculation 100 0.5 0.3 = 105 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_stock_price_problem_l919_91972


namespace NUMINAMATH_CALUDE_satisfying_function_is_constant_l919_91964

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℕ → ℤ) : Prop :=
  (∀ a b : ℕ, a > 0 ∧ b > 0 → a ∣ b → f a ≥ f b) ∧
  (∀ a b : ℕ, a > 0 ∧ b > 0 → f (a * b) + f (a^2 + b^2) = f a + f b)

/-- The main theorem stating that any satisfying function is constant -/
theorem satisfying_function_is_constant (f : ℕ → ℤ) (hf : SatisfyingFunction f) :
  ∃ C : ℤ, ∀ n : ℕ, f n = C :=
sorry

end NUMINAMATH_CALUDE_satisfying_function_is_constant_l919_91964


namespace NUMINAMATH_CALUDE_perfect_power_arithmetic_sequence_l919_91977

/-- A perfect power is a number that can be expressed as an integer raised to a positive integer exponent. -/
def IsPerfectPower (x : ℕ) : Prop :=
  ∃ (b e : ℕ), e > 0 ∧ x = b ^ e

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def IsArithmeticSequence (s : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ i, i < n → s i = a + i * d

/-- A sequence is non-constant if it has at least two distinct terms. -/
def IsNonConstant (s : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ i j, i < n ∧ j < n ∧ i ≠ j ∧ s i ≠ s j

/-- For any positive integer n, there exists a non-constant arithmetic sequence of length n
    where all terms are perfect powers. -/
theorem perfect_power_arithmetic_sequence (n : ℕ) (hn : n > 0) :
  ∃ (s : ℕ → ℕ),
    IsArithmeticSequence s n ∧
    IsNonConstant s n ∧
    (∀ i, i < n → IsPerfectPower (s i)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_power_arithmetic_sequence_l919_91977


namespace NUMINAMATH_CALUDE_problem_solution_l919_91953

def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1|

theorem problem_solution (m : ℝ) (a b c : ℝ) 
  (h1 : m > 0)
  (h2 : Set.Icc (-3 : ℝ) 3 = {x : ℝ | f m (x + 1) ≥ 0})
  (h3 : a > 0 ∧ b > 0 ∧ c > 0)
  (h4 : 1/a + 1/(2*b) + 1/(3*c) = m) :
  m = 3 ∧ a + 2*b + 3*c ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l919_91953


namespace NUMINAMATH_CALUDE_unique_zero_point_implies_same_sign_l919_91989

theorem unique_zero_point_implies_same_sign (f : ℝ → ℝ) :
  Continuous f →
  (∃! x, x ∈ (Set.Ioo 0 2) ∧ f x = 0) →
  f 2 * f 16 > 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_point_implies_same_sign_l919_91989


namespace NUMINAMATH_CALUDE_shorter_diagonal_is_25_l919_91925

/-- Represents a trapezoid EFGH -/
structure Trapezoid where
  ef : ℝ  -- length of side EF
  gh : ℝ  -- length of side GH
  eg : ℝ  -- length of side EG
  fh : ℝ  -- length of side FH
  ef_parallel_gh : ef > gh  -- EF is parallel to GH and longer
  e_acute : True  -- angle E is acute
  f_acute : True  -- angle F is acute

/-- The length of the shorter diagonal of the trapezoid -/
def shorter_diagonal (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that for a trapezoid with given side lengths, the shorter diagonal is 25 -/
theorem shorter_diagonal_is_25 (t : Trapezoid) 
  (h1 : t.ef = 39) 
  (h2 : t.gh = 27) 
  (h3 : t.eg = 13) 
  (h4 : t.fh = 15) : 
  shorter_diagonal t = 25 := by sorry

end NUMINAMATH_CALUDE_shorter_diagonal_is_25_l919_91925


namespace NUMINAMATH_CALUDE_cone_lateral_area_l919_91946

/-- The lateral area of a cone with base radius 2 cm and height 1 cm is 2√5π cm² -/
theorem cone_lateral_area : 
  let base_radius : ℝ := 2
  let height : ℝ := 1
  let slant_height : ℝ := Real.sqrt (base_radius ^ 2 + height ^ 2)
  let lateral_area : ℝ := π * base_radius * slant_height
  lateral_area = 2 * Real.sqrt 5 * π := by sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l919_91946


namespace NUMINAMATH_CALUDE_angle4_value_l919_91941

-- Define the angles as real numbers
variable (angle1 angle2 angle3 angle4 : ℝ)

-- State the theorem
theorem angle4_value
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4) :
  angle4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_angle4_value_l919_91941


namespace NUMINAMATH_CALUDE_wheat_bags_weight_l919_91955

def standard_weight : ℕ := 150
def num_bags : ℕ := 10
def deviations : List ℤ := [-6, -3, -1, -2, 7, 3, 4, -3, -2, 1]

theorem wheat_bags_weight :
  (List.sum deviations = -2) ∧
  (num_bags * standard_weight + List.sum deviations = 1498) :=
sorry

end NUMINAMATH_CALUDE_wheat_bags_weight_l919_91955


namespace NUMINAMATH_CALUDE_kevins_prizes_l919_91956

theorem kevins_prizes (total_prizes stuffed_animals frisbees : ℕ) 
  (h1 : total_prizes = 50)
  (h2 : stuffed_animals = 14)
  (h3 : frisbees = 18) :
  total_prizes - (stuffed_animals + frisbees) = 18 := by
  sorry

end NUMINAMATH_CALUDE_kevins_prizes_l919_91956


namespace NUMINAMATH_CALUDE_least_positive_tan_value_l919_91936

theorem least_positive_tan_value (x a b : ℝ) (h1 : Real.tan x = a / b) 
  (h2 : Real.tan (2 * x) = 2 * b / (a + 2 * b)) :
  ∃ k, k > 0 ∧ x = k ∧ Real.arctan 1 = k := by sorry

end NUMINAMATH_CALUDE_least_positive_tan_value_l919_91936


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l919_91957

/-- Given a square and a circle intersecting such that each side of the square contains
    a chord of the circle with length equal to half the radius of the circle,
    the ratio of the area of the square to the area of the circle is 3/(4π). -/
theorem square_circle_area_ratio (r : ℝ) (h : r > 0) :
  let s := r * Real.sqrt 3 / 2
  (s^2) / (π * r^2) = 3 / (4 * π) := by sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l919_91957


namespace NUMINAMATH_CALUDE_smallest_valid_number_l919_91938

def contains_all_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ≥ 1 ∧ d ≤ 9 → ∃ k : ℕ, n / (10^k) % 10 = d

def is_smallest_valid_number (n : ℕ) : Prop :=
  n % 72 = 0 ∧
  contains_all_digits n ∧
  ∀ m : ℕ, m < n → ¬(m % 72 = 0 ∧ contains_all_digits m)

theorem smallest_valid_number : is_smallest_valid_number 123457968 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l919_91938


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l919_91919

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem smallest_composite_no_small_factors : 
  (∀ n < 289, is_composite n → ∃ p, p < 15 ∧ Nat.Prime p ∧ p ∣ n) ∧ 
  is_composite 289 ∧
  (∀ p, Nat.Prime p → p ∣ 289 → p ≥ 15) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l919_91919


namespace NUMINAMATH_CALUDE_frog_jump_probability_l919_91921

/-- The probability of reaching a vertical side when starting from a given point -/
def P (x y : ℕ) : ℚ :=
  sorry

/-- The square grid size -/
def gridSize : ℕ := 5

theorem frog_jump_probability :
  P 2 1 = 13 / 24 :=
by
  have h1 : ∀ x y, x = 0 ∨ x = gridSize → P x y = 1 := sorry
  have h2 : ∀ x y, y = 0 ∨ y = gridSize → P x y = 0 := sorry
  have h3 : ∀ x y, 0 < x ∧ x < gridSize ∧ 0 < y ∧ y < gridSize →
    P x y = (P (x-1) y + P (x+1) y + P x (y-1) + P x (y+1)) / 4 := sorry
  sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l919_91921


namespace NUMINAMATH_CALUDE_polynomial_simplification_l919_91958

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^5 + 3 * x^4 + x^3 - 5 * x + 2) + (x^5 - 3 * x^4 - 2 * x^3 + x^2 + 5 * x - 7) = 
  3 * x^5 - x^3 + x^2 - 5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l919_91958


namespace NUMINAMATH_CALUDE_geometric_series_relation_l919_91979

/-- Given two infinite geometric series with specific conditions, prove that m = 7 -/
theorem geometric_series_relation (m : ℤ) : 
  let a₁ : ℚ := 15  -- first term of both series
  let b₁ : ℚ := 5   -- second term of first series
  let b₂ : ℚ := 5 + m  -- second term of second series
  let r₁ : ℚ := b₁ / a₁  -- common ratio of first series
  let r₂ : ℚ := b₂ / a₁  -- common ratio of second series
  let S₁ : ℚ := a₁ / (1 - r₁)  -- sum of first series
  let S₂ : ℚ := a₁ / (1 - r₂)  -- sum of second series
  S₂ = 3 * S₁ → m = 7 := by
  sorry


end NUMINAMATH_CALUDE_geometric_series_relation_l919_91979


namespace NUMINAMATH_CALUDE_total_insect_legs_l919_91954

/-- Given the number of insects and legs per insect, calculate the total number of insect legs -/
theorem total_insect_legs (num_insects : ℕ) (legs_per_insect : ℕ) :
  num_insects * legs_per_insect = num_insects * legs_per_insect := by
  sorry

/-- Ezekiel's report on insects in the laboratory -/
def ezekiels_report : ℕ := 9

/-- Number of legs each insect has -/
def legs_per_insect : ℕ := 6

/-- Calculate the total number of insect legs in the laboratory -/
def total_legs : ℕ := ezekiels_report * legs_per_insect

#eval total_legs  -- This will output 54

end NUMINAMATH_CALUDE_total_insect_legs_l919_91954


namespace NUMINAMATH_CALUDE_prob_both_blue_l919_91909

/-- Represents a jar containing buttons -/
structure Jar :=
  (red : ℕ)
  (blue : ℕ)

/-- The probability of selecting a blue button from a jar -/
def prob_blue (j : Jar) : ℚ :=
  j.blue / (j.red + j.blue)

/-- The initial state of Jar C -/
def initial_jar_c : Jar :=
  ⟨6, 12⟩

/-- The number of buttons moved from Jar C to Jar D -/
def buttons_moved : ℕ := 6

/-- The final state of Jar C after moving buttons -/
def final_jar_c : Jar :=
  ⟨initial_jar_c.red - buttons_moved / 2, initial_jar_c.blue - buttons_moved / 2⟩

/-- The state of Jar D after receiving buttons -/
def jar_d : Jar :=
  ⟨buttons_moved / 2, buttons_moved / 2⟩

theorem prob_both_blue :
  prob_blue final_jar_c * prob_blue jar_d = 3 / 8 :=
sorry

end NUMINAMATH_CALUDE_prob_both_blue_l919_91909


namespace NUMINAMATH_CALUDE_child_ticket_cost_child_ticket_cost_is_9_l919_91910

theorem child_ticket_cost (adult_price : ℕ) (total_people : ℕ) (total_revenue : ℕ) (children : ℕ) : ℕ :=
  let adults := total_people - children
  let child_price := (total_revenue - adult_price * adults) / children
  child_price

theorem child_ticket_cost_is_9 :
  child_ticket_cost 16 24 258 18 = 9 := by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_child_ticket_cost_is_9_l919_91910


namespace NUMINAMATH_CALUDE_non_sophomore_musicians_count_l919_91995

/-- Represents the number of students who play a musical instrument in a college -/
structure MusicianCount where
  total : ℕ
  sophomore_play_percent : ℚ
  non_sophomore_not_play_percent : ℚ
  total_not_play_percent : ℚ

/-- Calculates the number of non-sophomores who play a musical instrument -/
def non_sophomore_musicians (mc : MusicianCount) : ℕ :=
  sorry

/-- Theorem stating the number of non-sophomores who play a musical instrument -/
theorem non_sophomore_musicians_count (mc : MusicianCount) 
  (h1 : mc.total = 400)
  (h2 : mc.sophomore_play_percent = 1/2)
  (h3 : mc.non_sophomore_not_play_percent = 2/5)
  (h4 : mc.total_not_play_percent = 11/25) :
  non_sophomore_musicians mc = 144 := by
  sorry

end NUMINAMATH_CALUDE_non_sophomore_musicians_count_l919_91995


namespace NUMINAMATH_CALUDE_baseball_cap_production_l919_91920

/-- Proves that given the conditions of the baseball cap factory problem, 
    the number of caps made in the third week is 300. -/
theorem baseball_cap_production : 
  ∀ (x : ℕ), 
    (320 + 400 + x + (320 + 400 + x) / 3 = 1360) → 
    x = 300 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cap_production_l919_91920


namespace NUMINAMATH_CALUDE_courtyard_length_l919_91978

/-- Proves that a rectangular courtyard with given dimensions and number of bricks has a specific length -/
theorem courtyard_length (width : ℝ) (brick_length : ℝ) (brick_width : ℝ) (num_bricks : ℕ) :
  width = 16 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  num_bricks = 20000 →
  (width * (num_bricks * brick_length * brick_width / width)) = 25 := by
  sorry

#check courtyard_length

end NUMINAMATH_CALUDE_courtyard_length_l919_91978


namespace NUMINAMATH_CALUDE_smallest_n_for_integral_solutions_l919_91931

theorem smallest_n_for_integral_solutions : 
  ∀ n : ℕ+, 
  (∃ x : ℤ, 12 * x^2 - n * x + 576 = 0) → 
  n ≥ 168 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_integral_solutions_l919_91931


namespace NUMINAMATH_CALUDE_division_problem_l919_91965

theorem division_problem (divisor : ℕ) : 
  (83 / divisor = 9) ∧ (83 % divisor = 2) → divisor = 9 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l919_91965


namespace NUMINAMATH_CALUDE_triangular_prism_sum_l919_91960

/-- A triangular prism is a three-dimensional shape with two triangular bases and three rectangular faces. -/
structure TriangularPrism where
  -- We don't need to define the internal structure, just the concept

/-- The number of faces in a triangular prism -/
def num_faces (prism : TriangularPrism) : ℕ := 5

/-- The number of edges in a triangular prism -/
def num_edges (prism : TriangularPrism) : ℕ := 9

/-- The number of vertices in a triangular prism -/
def num_vertices (prism : TriangularPrism) : ℕ := 6

/-- Theorem: The sum of the number of faces, edges, and vertices of a triangular prism is 20 -/
theorem triangular_prism_sum (prism : TriangularPrism) :
  num_faces prism + num_edges prism + num_vertices prism = 20 := by
  sorry

end NUMINAMATH_CALUDE_triangular_prism_sum_l919_91960


namespace NUMINAMATH_CALUDE_range_of_function_l919_91926

theorem range_of_function (x : ℝ) : -13 ≤ 5 * Real.sin x - 12 * Real.cos x ∧ 
                                     5 * Real.sin x - 12 * Real.cos x ≤ 13 := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l919_91926


namespace NUMINAMATH_CALUDE_exists_integer_between_sqrt2_and_sqrt11_l919_91944

theorem exists_integer_between_sqrt2_and_sqrt11 :
  ∃ m : ℤ, Real.sqrt 2 < m ∧ m < Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_between_sqrt2_and_sqrt11_l919_91944


namespace NUMINAMATH_CALUDE_necklace_count_l919_91922

/-- Represents the number of beads of each color available -/
structure BeadInventory where
  green : ℕ
  white : ℕ
  orange : ℕ

/-- Represents the number of beads of each color required for one necklace -/
structure NecklacePattern where
  green : ℕ
  white : ℕ
  orange : ℕ

/-- Calculates the maximum number of complete necklaces that can be created -/
def maxNecklaces (inventory : BeadInventory) (pattern : NecklacePattern) : ℕ :=
  min (inventory.green / pattern.green)
      (min (inventory.white / pattern.white)
           (inventory.orange / pattern.orange))

theorem necklace_count 
  (inventory : BeadInventory)
  (pattern : NecklacePattern)
  (h_inventory : inventory = { green := 200, white := 100, orange := 50 })
  (h_pattern : pattern = { green := 3, white := 1, orange := 1 }) :
  maxNecklaces inventory pattern = 50 := by
  sorry

#eval maxNecklaces { green := 200, white := 100, orange := 50 } { green := 3, white := 1, orange := 1 }

end NUMINAMATH_CALUDE_necklace_count_l919_91922


namespace NUMINAMATH_CALUDE_min_value_quadratic_l919_91905

theorem min_value_quadratic :
  (∃ (x : ℝ), x^2 + 12*x = -36) ∧ (∀ (x : ℝ), x^2 + 12*x ≥ -36) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l919_91905


namespace NUMINAMATH_CALUDE_arc_length_for_unit_angle_l919_91929

/-- Given a circle where the chord length corresponding to a central angle of 1 radian is 2,
    prove that the arc length corresponding to this central angle is 1/sin(1/2). -/
theorem arc_length_for_unit_angle (r : ℝ) : 
  (2 * r * Real.sin (1 / 2) = 2) → 
  (r * 1 = 1 / Real.sin (1 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_arc_length_for_unit_angle_l919_91929
