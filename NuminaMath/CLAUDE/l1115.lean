import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l1115_111584

theorem quadratic_equation_proof (k : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x^2 + (2*k - 1)*x + k^2 - 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁^2 + x₂^2 = 16 + x₁*x₂) →
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l1115_111584


namespace NUMINAMATH_CALUDE_x_value_l1115_111520

theorem x_value : ∃ x : ℚ, (3 * x - 2) / 7 = 15 ∧ x = 107 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1115_111520


namespace NUMINAMATH_CALUDE_max_product_sum_l1115_111524

theorem max_product_sum (X Y Z : ℕ) (sum_constraint : X + Y + Z = 15) :
  (∀ X' Y' Z' : ℕ, X' + Y' + Z' = 15 → 
    X' * Y' * Z' + X' * Y' + Y' * Z' + Z' * X' ≤ X * Y * Z + X * Y + Y * Z + Z * X) →
  X * Y * Z + X * Y + Y * Z + Z * X = 200 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_l1115_111524


namespace NUMINAMATH_CALUDE_gas_pressure_volume_relationship_l1115_111538

/-- Given a gas with initial pressure P1, initial volume V1, and final volume V2,
    where pressure and volume are inversely proportional at constant temperature,
    prove that the final pressure P2 is equal to (P1 * V1) / V2. -/
theorem gas_pressure_volume_relationship (P1 V1 V2 : ℝ) (h1 : P1 > 0) (h2 : V1 > 0) (h3 : V2 > 0) :
  let P2 := (P1 * V1) / V2
  ∀ k : ℝ, (P1 * V1 = k ∧ P2 * V2 = k) → P2 = (P1 * V1) / V2 := by
sorry

end NUMINAMATH_CALUDE_gas_pressure_volume_relationship_l1115_111538


namespace NUMINAMATH_CALUDE_car_features_l1115_111526

theorem car_features (total : ℕ) (steering : ℕ) (windows : ℕ) (both : ℕ) :
  total = 65 →
  steering = 45 →
  windows = 25 →
  both = 17 →
  total - (steering + windows - both) = 12 := by
sorry

end NUMINAMATH_CALUDE_car_features_l1115_111526


namespace NUMINAMATH_CALUDE_sports_lottery_winners_l1115_111590

theorem sports_lottery_winners
  (win : Prop → Prop)
  (A B C D : Prop)
  (h1 : A → B)
  (h2 : B → (C ∨ ¬A))
  (h3 : ¬D → (A ∧ ¬C))
  (h4 : D → A) :
  A ∧ B ∧ C ∧ D :=
by sorry

end NUMINAMATH_CALUDE_sports_lottery_winners_l1115_111590


namespace NUMINAMATH_CALUDE_average_matches_is_four_l1115_111593

/-- Represents the distribution of matches played in a badminton club --/
structure MatchDistribution :=
  (one_match : Nat)
  (two_matches : Nat)
  (four_matches : Nat)
  (six_matches : Nat)
  (eight_matches : Nat)

/-- Calculates the average number of matches played, rounded to the nearest whole number --/
def averageMatchesPlayed (d : MatchDistribution) : Nat :=
  let totalMatches := d.one_match * 1 + d.two_matches * 2 + d.four_matches * 4 + d.six_matches * 6 + d.eight_matches * 8
  let totalPlayers := d.one_match + d.two_matches + d.four_matches + d.six_matches + d.eight_matches
  let average := totalMatches / totalPlayers
  if totalMatches % totalPlayers >= totalPlayers / 2 then average + 1 else average

/-- The specific distribution of matches in the badminton club --/
def clubDistribution : MatchDistribution :=
  { one_match := 4
  , two_matches := 3
  , four_matches := 2
  , six_matches := 2
  , eight_matches := 8 }

theorem average_matches_is_four :
  averageMatchesPlayed clubDistribution = 4 := by sorry

end NUMINAMATH_CALUDE_average_matches_is_four_l1115_111593


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_conditions_l1115_111586

theorem greatest_integer_satisfying_conditions : ∃ (n : ℕ), 
  n < 150 ∧ 
  (∃ (a : ℕ), n = 9 * a - 2) ∧ 
  (∃ (b : ℕ), n = 11 * b - 4) ∧ 
  (∃ (c : ℕ), n = 5 * c + 1) ∧ 
  (∀ (m : ℕ), m < 150 → 
    (∃ (a' : ℕ), m = 9 * a' - 2) → 
    (∃ (b' : ℕ), m = 11 * b' - 4) → 
    (∃ (c' : ℕ), m = 5 * c' + 1) → 
    m ≤ n) ∧
  n = 142 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_conditions_l1115_111586


namespace NUMINAMATH_CALUDE_vector_perpendicular_l1115_111598

theorem vector_perpendicular (a b : ℝ × ℝ) :
  a = (2, 0) →
  b = (-1, 1) →
  b • (a + b) = 0 := by
sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l1115_111598


namespace NUMINAMATH_CALUDE_min_stones_to_remove_is_ten_l1115_111582

/-- Represents a chessboard configuration -/
def Chessboard := Fin 7 → Fin 8 → Bool

/-- Checks if there are five adjacent stones in any direction -/
def hasFiveAdjacent (board : Chessboard) : Bool :=
  sorry

/-- Counts the number of stones on the board -/
def stoneCount (board : Chessboard) : Nat :=
  sorry

/-- The minimal number of stones that must be removed -/
def minStonesToRemove : Nat := 10

/-- Theorem stating the minimal number of stones to remove -/
theorem min_stones_to_remove_is_ten :
  ∀ (initial : Chessboard),
    stoneCount initial = 56 →
    ∀ (final : Chessboard),
      (¬ hasFiveAdjacent final) →
      (stoneCount initial - stoneCount final ≥ minStonesToRemove) ∧
      (∃ (optimal : Chessboard),
        (¬ hasFiveAdjacent optimal) ∧
        (stoneCount initial - stoneCount optimal = minStonesToRemove)) :=
  sorry

end NUMINAMATH_CALUDE_min_stones_to_remove_is_ten_l1115_111582


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_specific_intersection_l1115_111588

/-- The hyperbola C: x²/a² - y² = 1 (a > 0) -/
def C (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 = 1 ∧ a > 0

/-- The line l: x + y = 1 -/
def l (x y : ℝ) : Prop := x + y = 1

/-- P is the intersection point of line l and the y-axis -/
def P : ℝ × ℝ := (0, 1)

/-- A and B are distinct intersection points of C and l -/
def intersectionPoints (a : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ C a A.1 A.2 ∧ l A.1 A.2 ∧ C a B.1 B.2 ∧ l B.1 B.2

/-- PA = (5/12)PB -/
def vectorRelation (A B : ℝ × ℝ) : Prop :=
  (A.1 - P.1, A.2 - P.2) = (5/12 * (B.1 - P.1), 5/12 * (B.2 - P.2))

theorem hyperbola_line_intersection (a : ℝ) :
  intersectionPoints a ↔ (0 < a ∧ a < Real.sqrt 2 ∧ a ≠ 1) :=
sorry

theorem specific_intersection (a : ℝ) (A B : ℝ × ℝ) :
  C a A.1 A.2 ∧ l A.1 A.2 ∧ C a B.1 B.2 ∧ l B.1 B.2 ∧ vectorRelation A B →
  a = 17/13 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_specific_intersection_l1115_111588


namespace NUMINAMATH_CALUDE_player_A_can_win_l1115_111575

/-- Represents a game board with three rows --/
structure GameBoard :=
  (row1 : List ℤ)
  (row2 : List ℤ)
  (row3 : List ℤ)

/-- Represents a player in the game --/
inductive Player
  | A
  | B

/-- Defines a valid game board configuration --/
def ValidBoard (board : GameBoard) : Prop :=
  Odd board.row1.length ∧ 
  Odd board.row2.length ∧ 
  Odd board.row3.length

/-- Defines the game state --/
structure GameState :=
  (board : GameBoard)
  (currentPlayer : Player)

/-- Defines a game strategy for player A --/
def Strategy := GameState → ℕ → ℤ → GameState

/-- Theorem: Player A can always achieve the desired row sums --/
theorem player_A_can_win (initialBoard : GameBoard) (targetSum1 targetSum2 targetSum3 : ℤ) :
  ValidBoard initialBoard →
  ∃ (strategy : Strategy),
    (∀ (finalBoard : GameBoard),
      (finalBoard.row1.sum = targetSum1) ∧
      (finalBoard.row2.sum = targetSum2) ∧
      (finalBoard.row3.sum = targetSum3)) :=
sorry

end NUMINAMATH_CALUDE_player_A_can_win_l1115_111575


namespace NUMINAMATH_CALUDE_expression_equality_l1115_111545

theorem expression_equality : (2^5 * 9^2) / (8^2 * 3^5) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1115_111545


namespace NUMINAMATH_CALUDE_quadratic_roots_l1115_111531

theorem quadratic_roots (x : ℝ) : x * (x + 1) = 3 * (x + 1) ↔ x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1115_111531


namespace NUMINAMATH_CALUDE_equation_solution_l1115_111522

theorem equation_solution : 
  ∃ x : ℚ, (10 * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3 ∧ x = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1115_111522


namespace NUMINAMATH_CALUDE_inequalities_given_ordered_reals_l1115_111594

theorem inequalities_given_ordered_reals (a b c : ℝ) 
  (h1 : a > 0) (h2 : 0 > b) (h3 : b > c) : 
  (a / c > a / b) ∧ 
  ((a - b) / (a - c) > b / c) ∧ 
  (a - c ≥ 2 * Real.sqrt ((a - b) * (b - c))) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_given_ordered_reals_l1115_111594


namespace NUMINAMATH_CALUDE_yard_sale_books_theorem_l1115_111548

/-- Represents Melanie's book collection --/
structure BookCollection where
  initial_books : ℕ
  current_books : ℕ
  magazines : ℕ

/-- Calculates the number of books bought at the yard sale --/
def books_bought (collection : BookCollection) : ℕ :=
  collection.current_books - collection.initial_books

/-- Theorem stating that the number of books bought at the yard sale
    is the difference between current and initial book counts --/
theorem yard_sale_books_theorem (collection : BookCollection)
    (h1 : collection.initial_books = 83)
    (h2 : collection.current_books = 167)
    (h3 : collection.magazines = 57) :
    books_bought collection = 84 := by
  sorry

end NUMINAMATH_CALUDE_yard_sale_books_theorem_l1115_111548


namespace NUMINAMATH_CALUDE_sequence_properties_l1115_111527

-- Define the arithmetic sequence a_n and its sum S_n
def a (n : ℕ) : ℝ := sorry
def S (n : ℕ) : ℝ := sorry

-- Define the geometric sequence b_n and its sum T_n
def b (n : ℕ) : ℝ := sorry
def T (n : ℕ) : ℝ := sorry

-- State the theorem
theorem sequence_properties :
  (a 3 = 5 ∧ S 3 = 9) →
  (∀ n : ℕ, a n = 2 * n - 1) ∧
  (∃ q : ℝ, q > 0 ∧ b 3 = a 5 ∧ T 3 = 13 ∧
    ∀ n : ℕ, T n = (3^n - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1115_111527


namespace NUMINAMATH_CALUDE_probability_both_from_c_l1115_111543

structure Workshop where
  name : String
  quantity : Nat

def total_quantity (workshops : List Workshop) : Nat :=
  workshops.foldl (fun acc w => acc + w.quantity) 0

def sample_size : Nat := 6

def stratified_sample (w : Workshop) (total : Nat) : Nat :=
  w.quantity * sample_size / total

theorem probability_both_from_c (workshops : List Workshop) :
  let total := total_quantity workshops
  let c_workshop := workshops.find? (fun w => w.name = "C")
  match c_workshop with
  | some c =>
    let c_samples := stratified_sample c total
    (c_samples.choose 2) / (sample_size.choose 2) = 1 / 5
  | none => False
  := by sorry

end NUMINAMATH_CALUDE_probability_both_from_c_l1115_111543


namespace NUMINAMATH_CALUDE_triangle_circle_relation_l1115_111509

theorem triangle_circle_relation (α β γ s R r : ℝ) :
  -- Triangle angles
  0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π →
  -- Perimeter is 2s
  s > 0 →
  -- R is the radius of the circumscribed circle
  R > 0 →
  -- r is the radius of the inscribed circle
  r > 0 →
  -- The theorem
  4 * R^2 * Real.cos α * Real.cos β * Real.cos γ = s^2 - (r + 2*R)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circle_relation_l1115_111509


namespace NUMINAMATH_CALUDE_march_earnings_proof_l1115_111560

/-- Represents the earnings of a car salesman -/
def salesman_earnings (base_salary : ℕ) (commission_per_car : ℕ) (cars_sold : ℕ) : ℕ :=
  base_salary + commission_per_car * cars_sold

theorem march_earnings_proof (base_salary : ℕ) (commission_per_car : ℕ) (march_cars : ℕ) 
  (h1 : base_salary = 1000)
  (h2 : commission_per_car = 200)
  (h3 : salesman_earnings base_salary commission_per_car 15 = 2 * salesman_earnings base_salary commission_per_car march_cars) :
  salesman_earnings base_salary commission_per_car march_cars = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_march_earnings_proof_l1115_111560


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1115_111516

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 > 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1115_111516


namespace NUMINAMATH_CALUDE_polynomial_product_equality_l1115_111571

theorem polynomial_product_equality (x : ℝ) : 
  (1 + x^3) * (1 - 2*x + x^4) = 1 - 2*x + x^3 - x^4 + x^7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_equality_l1115_111571


namespace NUMINAMATH_CALUDE_solve_equation_for_x_l1115_111542

theorem solve_equation_for_x :
  ∃ X : ℝ, 1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * X)) = 1200.0000000000002 ∧ X = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_for_x_l1115_111542


namespace NUMINAMATH_CALUDE_linear_function_composition_l1115_111502

theorem linear_function_composition (a b : ℝ) :
  (∀ x y : ℝ, x < y → (a * x + b) < (a * y + b)) →
  (∀ x : ℝ, a * (a * x + b) + b = 4 * x - 1) →
  a = -2 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_function_composition_l1115_111502


namespace NUMINAMATH_CALUDE_harry_friday_speed_l1115_111577

-- Define Harry's running speeds throughout the week
def monday_speed : ℝ := 10
def tuesday_speed : ℝ := monday_speed * (1 - 0.3)
def wednesday_speed : ℝ := monday_speed * (1 + 0.5)
def thursday_speed : ℝ := wednesday_speed
def friday_speed : ℝ := thursday_speed * (1 + 0.6)

-- Theorem statement
theorem harry_friday_speed : friday_speed = 24 := by
  sorry

end NUMINAMATH_CALUDE_harry_friday_speed_l1115_111577


namespace NUMINAMATH_CALUDE_not_algorithm_quadratic_roots_l1115_111512

/-- Represents a statement that might be an algorithm --/
inductive Statement
  | travel_plan : Statement
  | linear_equation_steps : Statement
  | quadratic_equation_roots : Statement
  | sum_calculation : Statement

/-- Predicate to determine if a statement is an algorithm --/
def is_algorithm (s : Statement) : Prop :=
  match s with
  | Statement.travel_plan => True
  | Statement.linear_equation_steps => True
  | Statement.quadratic_equation_roots => False
  | Statement.sum_calculation => True

theorem not_algorithm_quadratic_roots :
  ¬(is_algorithm Statement.quadratic_equation_roots) ∧
  (is_algorithm Statement.travel_plan) ∧
  (is_algorithm Statement.linear_equation_steps) ∧
  (is_algorithm Statement.sum_calculation) := by
  sorry

end NUMINAMATH_CALUDE_not_algorithm_quadratic_roots_l1115_111512


namespace NUMINAMATH_CALUDE_no_function_satisfies_condition_l1115_111507

-- Define the function g
def g : ℝ → ℝ := sorry

-- Properties of g
axiom g_integer : ∀ n : ℤ, g n = (-1) ^ n
axiom g_affine : ∀ n : ℤ, ∀ x : ℝ, n ≤ x → x ≤ n + 1 → 
  ∃ a b : ℝ, ∀ y : ℝ, n ≤ y → y ≤ n + 1 → g y = a * y + b

-- Theorem statement
theorem no_function_satisfies_condition : 
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = f x + g y := by sorry

end NUMINAMATH_CALUDE_no_function_satisfies_condition_l1115_111507


namespace NUMINAMATH_CALUDE_blueberries_needed_for_pies_l1115_111599

-- Define the constants
def blueberries_per_pint : ℕ := 200
def pints_per_quart : ℕ := 2
def pies_to_make : ℕ := 6

-- Define the theorem
theorem blueberries_needed_for_pies : 
  blueberries_per_pint * pints_per_quart * pies_to_make = 2400 := by
  sorry

end NUMINAMATH_CALUDE_blueberries_needed_for_pies_l1115_111599


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l1115_111528

/-- 
Given a man and his son, where:
- The man is currently 30 years older than his son
- The son's present age is 28 years

This theorem proves that it will take 2 years for the man's age 
to be twice his son's age.
-/
theorem mans_age_twice_sons (
  son_age : ℕ) 
  (man_age : ℕ) 
  (h1 : son_age = 28) 
  (h2 : man_age = son_age + 30) : 
  ∃ (years : ℕ), years = 2 ∧ man_age + years = 2 * (son_age + years) :=
sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l1115_111528


namespace NUMINAMATH_CALUDE_quarter_circles_sum_limit_l1115_111500

/-- The sum of the lengths of quarter-circles approaches a value between the diameter and semi-circumference -/
theorem quarter_circles_sum_limit (D : ℝ) (h : D > 0) :
  ∃ (L : ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |2 * n * (π * D / (8 * n)) - L| < ε) ∧
             D < L ∧ L < π * D / 2 := by
  sorry

end NUMINAMATH_CALUDE_quarter_circles_sum_limit_l1115_111500


namespace NUMINAMATH_CALUDE_three_digit_multiples_of_6_and_8_l1115_111552

theorem three_digit_multiples_of_6_and_8 : 
  (Finset.filter (fun n => n % 6 = 0 ∧ n % 8 = 0) (Finset.range 900 ∪ {999})).card = 37 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_multiples_of_6_and_8_l1115_111552


namespace NUMINAMATH_CALUDE_parabola_line_theorem_l1115_111519

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop := x = m*y + 1

-- Define the perpendicular bisector of a line with slope m
def perpendicular_bisector (m : ℝ) (x y : ℝ) : Prop := x = -1/m * y + (2*m^2 + 3)

-- Define the condition for points to lie on the same circle
def on_same_circle (A B M N : ℝ × ℝ) : Prop := 
  let midpoint_AB := ((A.1 + B.1)/2, (A.2 + B.2)/2)
  let midpoint_MN := ((M.1 + N.1)/2, (M.2 + N.2)/2)
  (A.1 - midpoint_MN.1)^2 + (A.2 - midpoint_MN.2)^2 = 
  (M.1 - midpoint_AB.1)^2 + (M.2 - midpoint_AB.2)^2

theorem parabola_line_theorem (m : ℝ) :
  ∃ A B M N : ℝ × ℝ,
    parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ parabola M.1 M.2 ∧ parabola N.1 N.2 ∧
    line_through_focus m A.1 A.2 ∧ line_through_focus m B.1 B.2 ∧
    perpendicular_bisector m M.1 M.2 ∧ perpendicular_bisector m N.1 N.2 ∧
    on_same_circle A B M N →
  m = 1 ∨ m = -1 := by sorry

end NUMINAMATH_CALUDE_parabola_line_theorem_l1115_111519


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1115_111508

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 4 * a 8 = 4 →
  a 5 * a 6 * a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1115_111508


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1115_111569

-- Define the sets P and Q
def P : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def Q : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}

-- Theorem statement
theorem intersection_of_P_and_Q :
  P ∩ Q = {x : ℝ | 3 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1115_111569


namespace NUMINAMATH_CALUDE_no_finite_moves_to_fill_board_l1115_111564

-- Define the chessboard as a type
def Chessboard := ℤ × ℤ

-- Define the set A
def A : Set Chessboard :=
  {p | 100 ∣ p.1 ∧ 100 ∣ p.2}

-- Define a king's move
def is_valid_move (start finish : Chessboard) : Prop :=
  (start = finish) ∨
  (abs (start.1 - finish.1) ≤ 1 ∧ abs (start.2 - finish.2) ≤ 1)

-- Define the initial configuration of kings
def initial_kings : Set Chessboard :=
  {p | p ∉ A}

-- Define the state after k moves
def state_after_moves (k : ℕ) : Set Chessboard → Set Chessboard :=
  sorry

-- The main theorem
theorem no_finite_moves_to_fill_board :
  ¬ ∃ (k : ℕ), (state_after_moves k initial_kings) = Set.univ :=
sorry

end NUMINAMATH_CALUDE_no_finite_moves_to_fill_board_l1115_111564


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1115_111550

/-- Given a quadratic equation x^2 - (k+3)x + 2k + 2 = 0, prove:
    1. The equation always has two real roots
    2. When one root is positive and less than 1, -1 < k < 0 -/
theorem quadratic_equation_properties (k : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - (k+3)*x + 2*k + 2
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) ∧
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 → -1 < k ∧ k < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1115_111550


namespace NUMINAMATH_CALUDE_expression_value_l1115_111592

theorem expression_value (a b : ℚ) (ha : a = -1) (hb : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1115_111592


namespace NUMINAMATH_CALUDE_team_leader_deputy_count_l1115_111597

def people : Nat := 5

theorem team_leader_deputy_count : 
  (people * (people - 1) : Nat) = 20 := by
  sorry

end NUMINAMATH_CALUDE_team_leader_deputy_count_l1115_111597


namespace NUMINAMATH_CALUDE_minimum_soldiers_to_add_l1115_111551

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) :
  (84 - N % 84) = 82 :=
sorry

end NUMINAMATH_CALUDE_minimum_soldiers_to_add_l1115_111551


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1115_111573

theorem ellipse_eccentricity (a b x₀ y₀ : ℝ) (h₁ : a > b) (h₂ : b > 0) 
  (h₃ : x₀^2 / a^2 + y₀^2 / b^2 = 1) 
  (h₄ : y₀^2 / ((x₀ + a) * (a - x₀)) = 1/3) : 
  Real.sqrt (1 - b^2 / a^2) = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1115_111573


namespace NUMINAMATH_CALUDE_integer_pair_conditions_l1115_111567

theorem integer_pair_conditions (a b : ℕ+) : 
  (∃ k : ℕ, a^3 = k * b^2) ∧ 
  (∃ m : ℕ, b - 1 = m * (a - 1)) → 
  (a = b) ∨ (b = 1) := by
sorry

end NUMINAMATH_CALUDE_integer_pair_conditions_l1115_111567


namespace NUMINAMATH_CALUDE_expression_simplification_l1115_111532

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 - 1) :
  (a - 1) / (a^2 - 2*a + 1) / ((a^2 + a) / (a^2 - 1) + 1 / (a - 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1115_111532


namespace NUMINAMATH_CALUDE_log_equality_difference_l1115_111537

theorem log_equality_difference (a b c d : ℕ+) 
  (h1 : (Real.log b) / (Real.log a) = (Real.log d) / (Real.log c))
  (h2 : a - c = 9) : 
  b - d = 93 := by sorry

end NUMINAMATH_CALUDE_log_equality_difference_l1115_111537


namespace NUMINAMATH_CALUDE_book_distribution_ways_l1115_111514

/-- The number of ways to distribute identical books between two states --/
def distribute_books (n : ℕ) : ℕ := n - 1

/-- The number of books --/
def total_books : ℕ := 8

/-- Theorem: The number of ways to distribute eight identical books between
    the library and being checked out, with at least one book in each state,
    is equal to 7. --/
theorem book_distribution_ways :
  distribute_books total_books = 7 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_ways_l1115_111514


namespace NUMINAMATH_CALUDE_assignment_methods_eq_eight_l1115_111510

/-- Represents the number of schools --/
def num_schools : ℕ := 2

/-- Represents the number of student teachers --/
def num_teachers : ℕ := 4

/-- Calculates the number of assignment methods --/
def assignment_methods : ℕ := 
  let a_assignments := num_schools -- A can be assigned to either school
  let b_assignments := num_schools - 1 -- B must be assigned to the other school
  let remaining_assignments := num_schools ^ (num_teachers - 2) -- Remaining 2 teachers can be assigned freely
  a_assignments * b_assignments * remaining_assignments

/-- Theorem stating that the number of assignment methods is 8 --/
theorem assignment_methods_eq_eight : assignment_methods = 8 := by
  sorry

end NUMINAMATH_CALUDE_assignment_methods_eq_eight_l1115_111510


namespace NUMINAMATH_CALUDE_red_balls_count_l1115_111587

/-- Given a bag with 2400 balls of red, green, and blue colors,
    where the ratio of red:green:blue is 15:13:17,
    prove that the number of red balls is 795. -/
theorem red_balls_count (total : ℕ) (red green blue : ℕ) :
  total = 2400 →
  red + green + blue = 45 →
  red = 15 →
  green = 13 →
  blue = 17 →
  red * (total / (red + green + blue)) = 795 := by
  sorry


end NUMINAMATH_CALUDE_red_balls_count_l1115_111587


namespace NUMINAMATH_CALUDE_second_month_sale_l1115_111517

def average_sale : ℕ := 5900
def first_month : ℕ := 5921
def third_month : ℕ := 5568
def fourth_month : ℕ := 6088
def fifth_month : ℕ := 6433
def sixth_month : ℕ := 5922

theorem second_month_sale :
  ∃ (second_month : ℕ),
    second_month = 
      6 * average_sale - (first_month + third_month + fourth_month + fifth_month + sixth_month) ∧
    second_month = 5468 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_l1115_111517


namespace NUMINAMATH_CALUDE_detached_calculations_l1115_111595

theorem detached_calculations : 
  (78 * 12 - 531 = 405) ∧ 
  (32 * (69 - 54) = 480) ∧ 
  (58 / 2 * 16 = 464) ∧ 
  (352 / 8 / 4 = 11) := by
  sorry

end NUMINAMATH_CALUDE_detached_calculations_l1115_111595


namespace NUMINAMATH_CALUDE_coefficient_of_x8_in_expansion_l1115_111518

/-- The coefficient of x^8 in the expansion of (1 + 3x - 2x^2)^5 is -720 -/
theorem coefficient_of_x8_in_expansion : 
  let p : Polynomial ℤ := 1 + 3 * X - 2 * X^2
  let coeff := (p^5).coeff 8
  coeff = -720 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x8_in_expansion_l1115_111518


namespace NUMINAMATH_CALUDE_expression_value_l1115_111541

theorem expression_value (a b c : ℤ) :
  a = 18 ∧ b = 20 ∧ c = 22 →
  (a - (b - c)) - ((a - b) - c) = 44 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1115_111541


namespace NUMINAMATH_CALUDE_parabola_through_points_with_parallel_tangent_l1115_111568

/-- A parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The slope of the tangent line to the parabola at a given x-coordinate -/
def Parabola.tangent_slope (p : Parabola) (x : ℝ) : ℝ :=
  2 * p.a * x + p.b

/-- Theorem stating the conditions and the result to be proved -/
theorem parabola_through_points_with_parallel_tangent 
  (p : Parabola) 
  (h1 : p.y_coord 1 = 1) 
  (h2 : p.y_coord 2 = -1) 
  (h3 : p.tangent_slope 2 = 1) : 
  p.a = 3 ∧ p.b = -11 ∧ p.c = 9 := by
  sorry


end NUMINAMATH_CALUDE_parabola_through_points_with_parallel_tangent_l1115_111568


namespace NUMINAMATH_CALUDE_binomial_coefficient_28_7_l1115_111535

theorem binomial_coefficient_28_7 
  (h1 : Nat.choose 26 3 = 2600)
  (h2 : Nat.choose 26 4 = 14950)
  (h3 : Nat.choose 26 5 = 65780) : 
  Nat.choose 28 7 = 197340 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_28_7_l1115_111535


namespace NUMINAMATH_CALUDE_exists_valid_distribution_with_plate_B_size_l1115_111576

/-- Represents a distribution of balls across three plates -/
structure BallDistribution where
  plateA : List Nat
  plateB : List Nat
  plateC : List Nat

/-- Checks if a given distribution satisfies the problem conditions -/
def isValidDistribution (d : BallDistribution) : Prop :=
  let allBalls := d.plateA ++ d.plateB ++ d.plateC
  (∀ n ∈ allBalls, 1 ≤ n ∧ n ≤ 15) ∧ 
  (allBalls.length = 15) ∧
  (d.plateA.length ≥ 4 ∧ d.plateB.length ≥ 4 ∧ d.plateC.length ≥ 4) ∧
  ((d.plateA.sum : Rat) / d.plateA.length = 3) ∧
  ((d.plateB.sum : Rat) / d.plateB.length = 8) ∧
  ((d.plateC.sum : Rat) / d.plateC.length = 13)

/-- The main theorem to be proved -/
theorem exists_valid_distribution_with_plate_B_size :
  ∃ d : BallDistribution, isValidDistribution d ∧ (d.plateB.length = 7 ∨ d.plateB.length = 5) := by
  sorry


end NUMINAMATH_CALUDE_exists_valid_distribution_with_plate_B_size_l1115_111576


namespace NUMINAMATH_CALUDE_cookie_sales_proof_l1115_111540

/-- The number of homes in Neighborhood A -/
def homes_a : ℕ := 10

/-- The number of boxes each home in Neighborhood A buys -/
def boxes_per_home_a : ℕ := 2

/-- The number of boxes each home in Neighborhood B buys -/
def boxes_per_home_b : ℕ := 5

/-- The cost of each box of cookies in dollars -/
def cost_per_box : ℕ := 2

/-- The total sales in dollars from the better neighborhood -/
def better_sales : ℕ := 50

/-- The number of homes in Neighborhood B -/
def homes_b : ℕ := 5

theorem cookie_sales_proof : 
  homes_b * boxes_per_home_b * cost_per_box = better_sales ∧
  homes_b * boxes_per_home_b * cost_per_box > homes_a * boxes_per_home_a * cost_per_box :=
by sorry

end NUMINAMATH_CALUDE_cookie_sales_proof_l1115_111540


namespace NUMINAMATH_CALUDE_girls_from_clay_l1115_111544

/-- Represents a school in the science camp --/
inductive School
| Jonas
| Clay
| Maple

/-- Represents the gender of a student --/
inductive Gender
| Boy
| Girl

/-- Represents the distribution of students in the science camp --/
structure CampDistribution where
  total_students : ℕ
  total_boys : ℕ
  total_girls : ℕ
  jonas_students : ℕ
  clay_students : ℕ
  maple_students : ℕ
  jonas_boys : ℕ

/-- The actual distribution of students in the science camp --/
def camp_distribution : CampDistribution :=
  { total_students := 120
  , total_boys := 70
  , total_girls := 50
  , jonas_students := 50
  , clay_students := 40
  , maple_students := 30
  , jonas_boys := 30
  }

/-- Theorem stating that the number of girls from Clay Middle School is 10 --/
theorem girls_from_clay (d : CampDistribution) (h : d = camp_distribution) :
  ∃ (clay_girls : ℕ), clay_girls = 10 ∧
  clay_girls = d.clay_students - (d.total_boys - d.jonas_boys) :=
by sorry

end NUMINAMATH_CALUDE_girls_from_clay_l1115_111544


namespace NUMINAMATH_CALUDE_no_rational_solutions_l1115_111553

theorem no_rational_solutions (m : ℕ+) : ¬∃ (x : ℚ), m * x^2 + 40 * x + m = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solutions_l1115_111553


namespace NUMINAMATH_CALUDE_fourth_root_63504000_l1115_111503

theorem fourth_root_63504000 : 
  (63504000 : ℝ)^(1/4) = 2 * (2 : ℝ)^(1/2) * (3 : ℝ)^(1/2) * (11 : ℝ)^(1/4) * 10^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_63504000_l1115_111503


namespace NUMINAMATH_CALUDE_largest_valid_domain_l1115_111562

def is_valid_domain (S : Set ℝ) : Prop :=
  ∃ g : ℝ → ℝ, 
    (∀ x ∈ S, (1 / x) ∈ S) ∧ 
    (∀ x ∈ S, g x + g (1 / x) = x^2)

theorem largest_valid_domain : 
  is_valid_domain {-1, 1} ∧ 
  ∀ S : Set ℝ, is_valid_domain S → S ⊆ {-1, 1} :=
sorry

end NUMINAMATH_CALUDE_largest_valid_domain_l1115_111562


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1115_111559

/-- Given a principal amount P and an unknown interest rate R, 
    if increasing the rate by 5% for 9 years results in $1,350 more interest,
    then the principal P must be $3,000. -/
theorem simple_interest_problem (P R : ℝ) : 
  P * (R + 5) * 9 / 100 - P * R * 9 / 100 = 1350 → P = 3000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1115_111559


namespace NUMINAMATH_CALUDE_sequence_pattern_l1115_111515

def S : ℕ → ℕ
  | n => if n % 2 = 1 then 1 + 2 * ((n + 1) / 2 - 1) else 2^(n / 2)

theorem sequence_pattern (n : ℕ) : 
  S n = if n % 2 = 1 then 1 + 2 * ((n + 1) / 2 - 1) else 2^(n / 2) := by
  sorry

#eval [S 1, S 2, S 3, S 4, S 5, S 6, S 7, S 8, S 9, S 10, S 11, S 12, S 13, S 14]

end NUMINAMATH_CALUDE_sequence_pattern_l1115_111515


namespace NUMINAMATH_CALUDE_complex_multiplication_l1115_111555

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (2 - i) = 1 + 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1115_111555


namespace NUMINAMATH_CALUDE_chef_potato_usage_l1115_111574

/-- The number of potatoes used for lunch -/
def lunch_potatoes : ℕ := 5

/-- The number of potatoes used for dinner -/
def dinner_potatoes : ℕ := 2

/-- The total number of potatoes used -/
def total_potatoes : ℕ := lunch_potatoes + dinner_potatoes

theorem chef_potato_usage : total_potatoes = 7 := by
  sorry

end NUMINAMATH_CALUDE_chef_potato_usage_l1115_111574


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1115_111596

theorem polynomial_factorization :
  (∀ x : ℝ, 2 * x^4 - 2 = 2 * (x^2 + 1) * (x + 1) * (x - 1)) ∧
  (∀ x : ℝ, x^4 - 18 * x^2 + 81 = (x + 3)^2 * (x - 3)^2) ∧
  (∀ y : ℝ, (y^2 - 1)^2 + 11 * (1 - y^2) + 24 = (y + 2) * (y - 2) * (y + 3) * (y - 3)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1115_111596


namespace NUMINAMATH_CALUDE_eli_age_difference_l1115_111547

/-- Given the ages and relationships between Kaylin, Sarah, Eli, and Freyja, prove that Eli is 9 years older than Freyja. -/
theorem eli_age_difference (kaylin sarah eli freyja : ℕ) : 
  kaylin = 33 →
  freyja = 10 →
  sarah = kaylin + 5 →
  sarah = 2 * eli →
  eli > freyja →
  eli - freyja = 9 := by
  sorry

end NUMINAMATH_CALUDE_eli_age_difference_l1115_111547


namespace NUMINAMATH_CALUDE_sum_f_positive_l1115_111591

noncomputable def f (x : ℝ) : ℝ := x^3 / Real.cos x

theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
  (h₁ : |x₁| < π/2) (h₂ : |x₂| < π/2) (h₃ : |x₃| < π/2)
  (h₄ : x₁ + x₂ > 0) (h₅ : x₂ + x₃ > 0) (h₆ : x₁ + x₃ > 0) :
  f x₁ + f x₂ + f x₃ > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_positive_l1115_111591


namespace NUMINAMATH_CALUDE_sum_of_real_and_imaginary_parts_of_one_plus_i_squared_l1115_111572

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_real_and_imaginary_parts_of_one_plus_i_squared (a b : ℝ) : 
  (1 + i)^2 = a + b * i → a + b = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_real_and_imaginary_parts_of_one_plus_i_squared_l1115_111572


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l1115_111546

theorem quadratic_form_minimum (x y : ℝ) :
  3 * x^2 + 4 * x * y + y^2 - 6 * x + 2 * y + 9 ≥ 9/4 ∧
  (3 * x^2 + 4 * x * y + y^2 - 6 * x + 2 * y + 9 = 9/4 ↔ x = 3/2 ∧ y = -3/4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l1115_111546


namespace NUMINAMATH_CALUDE_time_after_1500_seconds_l1115_111579

-- Define the initial time
def initial_time : Nat × Nat := (14, 35)

-- Define the elapsed time in seconds
def elapsed_seconds : Nat := 1500

-- Function to add seconds to a time
def add_seconds (time : Nat × Nat) (seconds : Nat) : Nat × Nat :=
  sorry

-- Theorem to prove
theorem time_after_1500_seconds : 
  add_seconds initial_time elapsed_seconds = (15, 0) := by
  sorry

end NUMINAMATH_CALUDE_time_after_1500_seconds_l1115_111579


namespace NUMINAMATH_CALUDE_count_integer_values_is_ten_l1115_111505

/-- The number of integer values of n for which 8000 * (2/5)^n is an integer --/
def count_integer_values : ℕ := 10

/-- Predicate to check if a given number is an integer --/
def is_integer (x : ℚ) : Prop := ∃ (k : ℤ), x = k

/-- The main theorem stating that there are exactly 10 integer values of n
    for which 8000 * (2/5)^n is an integer --/
theorem count_integer_values_is_ten :
  (∃! (s : Finset ℤ), s.card = count_integer_values ∧
    ∀ n : ℤ, n ∈ s ↔ is_integer (8000 * (2/5)^n)) :=
sorry

end NUMINAMATH_CALUDE_count_integer_values_is_ten_l1115_111505


namespace NUMINAMATH_CALUDE_tangent_line_to_two_parabolas_l1115_111513

/-- Given curves C₁: y = x² and C₂: y = -(x - 2)², prove that the line l: y = -2x + 3 is tangent to both C₁ and C₂ -/
theorem tangent_line_to_two_parabolas :
  let C₁ : ℝ → ℝ := λ x ↦ x^2
  let C₂ : ℝ → ℝ := λ x ↦ -(x - 2)^2
  let l : ℝ → ℝ := λ x ↦ -2*x + 3
  (∃ x₁, (C₁ x₁ = l x₁) ∧ (deriv C₁ x₁ = deriv l x₁)) ∧
  (∃ x₂, (C₂ x₂ = l x₂) ∧ (deriv C₂ x₂ = deriv l x₂)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_two_parabolas_l1115_111513


namespace NUMINAMATH_CALUDE_fifteen_plus_sixteen_l1115_111578

theorem fifteen_plus_sixteen : 15 + 16 = 31 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_plus_sixteen_l1115_111578


namespace NUMINAMATH_CALUDE_passengers_landed_late_l1115_111533

theorem passengers_landed_late (on_time passengers_landed_on_time : ℕ) 
  (total passengers_landed_total : ℕ) 
  (h1 : passengers_landed_on_time = 14507) 
  (h2 : passengers_landed_total = 14720) : 
  passengers_landed_total - passengers_landed_on_time = 213 := by
  sorry

end NUMINAMATH_CALUDE_passengers_landed_late_l1115_111533


namespace NUMINAMATH_CALUDE_parity_of_A_15_16_17_l1115_111501

def A : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => A (n + 2) + A n

theorem parity_of_A_15_16_17 : 
  Odd (A 15) ∧ Even (A 16) ∧ Odd (A 17) := by sorry

end NUMINAMATH_CALUDE_parity_of_A_15_16_17_l1115_111501


namespace NUMINAMATH_CALUDE_last_page_cards_l1115_111549

/-- Represents the number of cards that can be placed on different page types -/
inductive PageType
| Four : PageType
| Six : PageType
| Eight : PageType

/-- Calculates the number of cards on the last partially-filled page -/
def cardsOnLastPage (totalCards : ℕ) (pageTypes : List PageType) : ℕ :=
  sorry

/-- Theorem stating that for 137 cards and the given page types, 
    the number of cards on the last partially-filled page is 1 -/
theorem last_page_cards : 
  cardsOnLastPage 137 [PageType.Four, PageType.Six, PageType.Eight] = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_page_cards_l1115_111549


namespace NUMINAMATH_CALUDE_sector_radius_l1115_111539

/-- Given a circular sector with area 10 cm² and arc length 4 cm, prove that the radius is 5 cm -/
theorem sector_radius (area : ℝ) (arc_length : ℝ) (radius : ℝ) 
  (h_area : area = 10) 
  (h_arc : arc_length = 4) 
  (h_sector : area = (arc_length * radius) / 2) : radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l1115_111539


namespace NUMINAMATH_CALUDE_find_divisor_l1115_111589

theorem find_divisor (x y n : ℕ+) : 
  x = n * y + 4 →
  2 * x = 14 * y + 1 →
  5 * y - x = 3 →
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l1115_111589


namespace NUMINAMATH_CALUDE_systematic_sample_result_l1115_111557

def systematic_sample (total : ℕ) (sample_size : ℕ) (interval_start : ℕ) (interval_end : ℕ) : ℕ :=
  let sample_interval := total / sample_size
  let interval_size := interval_end - interval_start + 1
  interval_size / sample_interval

theorem systematic_sample_result :
  systematic_sample 360 20 181 288 = 6 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_result_l1115_111557


namespace NUMINAMATH_CALUDE_election_winner_percentage_l1115_111506

theorem election_winner_percentage : 
  ∀ (total_votes : ℕ) (winner_votes loser_votes : ℕ),
  winner_votes = 864 →
  winner_votes - loser_votes = 288 →
  total_votes = winner_votes + loser_votes →
  (winner_votes : ℚ) / (total_votes : ℚ) = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l1115_111506


namespace NUMINAMATH_CALUDE_brand_z_percentage_l1115_111583

theorem brand_z_percentage (tank_capacity : ℝ) (brand_z_amount : ℝ) (brand_x_amount : ℝ)
  (h1 : tank_capacity > 0)
  (h2 : brand_z_amount = 1/8 * tank_capacity)
  (h3 : brand_x_amount = 7/8 * tank_capacity)
  (h4 : brand_z_amount + brand_x_amount = tank_capacity) :
  (brand_z_amount / tank_capacity) * 100 = 12.5 := by
sorry

end NUMINAMATH_CALUDE_brand_z_percentage_l1115_111583


namespace NUMINAMATH_CALUDE_frank_lamp_purchase_l1115_111585

/-- Frank's lamp purchase problem -/
theorem frank_lamp_purchase (frank_money : ℕ) (cheapest_lamp : ℕ) (expensive_factor : ℕ) :
  frank_money = 90 →
  cheapest_lamp = 20 →
  expensive_factor = 3 →
  frank_money - (cheapest_lamp * expensive_factor) = 30 := by
  sorry

end NUMINAMATH_CALUDE_frank_lamp_purchase_l1115_111585


namespace NUMINAMATH_CALUDE_unique_solution_for_system_l1115_111529

theorem unique_solution_for_system : ∃! y : ℚ, 9 * y^2 + 8 * y - 2 = 0 ∧ 27 * y^2 + 62 * y - 8 = 0 :=
  by sorry

end NUMINAMATH_CALUDE_unique_solution_for_system_l1115_111529


namespace NUMINAMATH_CALUDE_income_difference_is_negative_150_l1115_111581

/-- Calculates the difference in income between Janet's first month as a freelancer and her current job -/
def income_difference : ℤ :=
  let current_job_weekly_hours : ℕ := 40
  let current_job_hourly_rate : ℕ := 30
  let freelance_weeks : List ℕ := [30, 35, 40, 50]
  let freelance_rates : List ℕ := [45, 40, 35, 38]
  let extra_fica_tax_weekly : ℕ := 25
  let healthcare_premium_monthly : ℕ := 400
  let increased_rent_monthly : ℕ := 750
  let business_expenses_monthly : ℕ := 150
  let weeks_per_month : ℕ := 4

  let current_job_monthly_income := current_job_weekly_hours * current_job_hourly_rate * weeks_per_month
  
  let freelance_monthly_income := (List.zip freelance_weeks freelance_rates).map (fun (h, r) => h * r) |>.sum
  
  let extra_expenses_monthly := extra_fica_tax_weekly * weeks_per_month + 
                                healthcare_premium_monthly + 
                                increased_rent_monthly + 
                                business_expenses_monthly
  
  let freelance_net_income := freelance_monthly_income - extra_expenses_monthly
  
  freelance_net_income - current_job_monthly_income

theorem income_difference_is_negative_150 : income_difference = -150 := by
  sorry

end NUMINAMATH_CALUDE_income_difference_is_negative_150_l1115_111581


namespace NUMINAMATH_CALUDE_salt_solution_replacement_l1115_111554

/-- Given two solutions with different salt concentrations, prove the fraction of
    the first solution replaced to achieve a specific final concentration -/
theorem salt_solution_replacement
  (initial_salt_concentration : Real)
  (second_salt_concentration : Real)
  (final_salt_concentration : Real)
  (h1 : initial_salt_concentration = 0.14)
  (h2 : second_salt_concentration = 0.22)
  (h3 : final_salt_concentration = 0.16) :
  ∃ (x : Real), 
    x = 1/4 ∧ 
    initial_salt_concentration + x * second_salt_concentration - 
      x * initial_salt_concentration = final_salt_concentration :=
by sorry

end NUMINAMATH_CALUDE_salt_solution_replacement_l1115_111554


namespace NUMINAMATH_CALUDE_total_pears_theorem_l1115_111521

/-- Calculates the total number of pears picked over three days given the number of pears picked by each person in one day -/
def total_pears_over_three_days (jason keith mike alicia tina nicola : ℕ) : ℕ :=
  3 * (jason + keith + mike + alicia + tina + nicola)

/-- Theorem stating that given the specific number of pears picked by each person,
    the total number of pears picked over three days is 654 -/
theorem total_pears_theorem :
  total_pears_over_three_days 46 47 12 28 33 52 = 654 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_theorem_l1115_111521


namespace NUMINAMATH_CALUDE_last_digit_power_sum_l1115_111525

theorem last_digit_power_sum (m : ℕ+) : (2^(m.val + 2007) + 2^(m.val + 1)) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_power_sum_l1115_111525


namespace NUMINAMATH_CALUDE_distribute_students_count_l1115_111504

/-- The number of ways to distribute 4 students among 3 universities --/
def distribute_students : ℕ :=
  -- We define the function without implementation
  sorry

/-- Theorem stating that the number of ways to distribute 4 students
    among 3 universities, with each university receiving at least 1 student,
    is equal to 36 --/
theorem distribute_students_count :
  distribute_students = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_students_count_l1115_111504


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l1115_111580

theorem inscribed_square_side_length (s : ℝ) : 
  s > 0 →                             -- s is positive (side length of smaller square)
  s^2 + (25 - s^2)/2 = 25 →           -- Area of smaller square plus half the difference equals area of larger square
  s = 5 * Real.sqrt 3 / 3 :=           -- Side length of smaller square is 5√3/3
by
  sorry


end NUMINAMATH_CALUDE_inscribed_square_side_length_l1115_111580


namespace NUMINAMATH_CALUDE_BRICS_is_set_closeToZero_is_not_l1115_111530

-- Define a type for countries
structure Country where
  name : String

-- Define the BRICS summit participants
def BRICS2016Participants : Set Country := sorry

-- Define a property for real numbers "close to 0"
def closeToZero (x : ℝ) : Prop := sorry

theorem BRICS_is_set_closeToZero_is_not :
  (∃ (S : Set Country), S = BRICS2016Participants) ∧
  (¬ ∃ (T : Set ℝ), ∀ x, x ∈ T ↔ closeToZero x) :=
sorry

end NUMINAMATH_CALUDE_BRICS_is_set_closeToZero_is_not_l1115_111530


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l1115_111563

/-- An arithmetic sequence with positive terms and non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : ∀ n, a n > 0
  h2 : d ≠ 0
  h3 : ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_inequality (seq : ArithmeticSequence) : 
  seq.a 1 * seq.a 8 < seq.a 4 * seq.a 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l1115_111563


namespace NUMINAMATH_CALUDE_intersection_and_union_of_sets_l1115_111565

theorem intersection_and_union_of_sets (x : ℝ) 
  (A : Set ℝ) (B : Set ℝ)
  (hA : A = {-3, x^2, x+1})
  (hB : B = {x-3, 2*x-1, x^2+1})
  (hIntersection : A ∩ B = {-3}) :
  x = -1 ∧ A ∪ B = {-4, -3, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_union_of_sets_l1115_111565


namespace NUMINAMATH_CALUDE_sin_690_degrees_l1115_111556

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_690_degrees_l1115_111556


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1115_111536

/-- Given two vectors are parallel if their coordinates are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v.1 = t * w.1 ∧ v.2 = t * w.2

theorem parallel_vectors_k_value :
  let e₁ : ℝ × ℝ := (1, 0)
  let e₂ : ℝ × ℝ := (0, 1)
  let a : ℝ × ℝ := (e₁.1 - 2 * e₂.1, e₁.2 - 2 * e₂.2)
  ∀ k : ℝ,
    let b : ℝ × ℝ := (k * e₁.1 + e₂.1, k * e₁.2 + e₂.2)
    are_parallel a b → k = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1115_111536


namespace NUMINAMATH_CALUDE_extended_morse_code_symbols_l1115_111534

def morse_symbols (n : Nat) : Nat :=
  3^n

theorem extended_morse_code_symbols :
  (morse_symbols 1) + (morse_symbols 2) + (morse_symbols 3) + (morse_symbols 4) = 120 := by
  sorry

end NUMINAMATH_CALUDE_extended_morse_code_symbols_l1115_111534


namespace NUMINAMATH_CALUDE_distinct_convex_polygons_l1115_111566

/-- Represents a triangle with side lengths --/
structure Triangle :=
  (side1 side2 side3 : ℝ)

/-- Represents a convex polygon --/
structure ConvexPolygon :=
  (vertices : List (ℝ × ℝ))

/-- Checks if a polygon is convex --/
def isConvex (p : ConvexPolygon) : Prop :=
  sorry

/-- Counts the number of distinct convex polygons that can be formed --/
def countConvexPolygons (triangles : List Triangle) : ℕ :=
  sorry

/-- The main theorem --/
theorem distinct_convex_polygons :
  let triangles : List Triangle := [
    ⟨3, 3, 3⟩, ⟨3, 3, 3⟩,  -- Two equilateral triangles
    ⟨3, 4, 5⟩, ⟨3, 4, 5⟩   -- Two scalene triangles
  ]
  countConvexPolygons triangles = 16 := by
  sorry

end NUMINAMATH_CALUDE_distinct_convex_polygons_l1115_111566


namespace NUMINAMATH_CALUDE_sticker_distribution_ways_l1115_111561

/-- The number of ways to distribute stickers across sheets of paper -/
def distribute_stickers (total_stickers : ℕ) (total_sheets : ℕ) : ℕ :=
  Nat.choose (total_stickers - total_sheets + total_sheets - 1) (total_sheets - 1)

/-- Theorem: There are 126 ways to distribute 10 stickers across 5 sheets -/
theorem sticker_distribution_ways :
  distribute_stickers 10 5 = 126 := by
  sorry

#eval distribute_stickers 10 5

end NUMINAMATH_CALUDE_sticker_distribution_ways_l1115_111561


namespace NUMINAMATH_CALUDE_square_area_with_line_area_of_square_ABCD_l1115_111511

/-- A square with a line passing through it -/
structure SquareWithLine where
  /-- The side length of the square -/
  side : ℝ
  /-- The distance from vertex A to the line -/
  dist_A : ℝ
  /-- The distance from vertex C to the line -/
  dist_C : ℝ
  /-- The line passes through the midpoint of AB -/
  midpoint_AB : dist_A = side / 2
  /-- The line intersects BC -/
  intersects_BC : dist_C < side

/-- The theorem stating the area of the square given the conditions -/
theorem square_area_with_line (s : SquareWithLine) (h1 : s.dist_A = 4) (h2 : s.dist_C = 7) : 
  s.side ^ 2 = 185 := by
  sorry

/-- The main theorem proving the area of the square ABCD is 185 -/
theorem area_of_square_ABCD : ∃ s : SquareWithLine, s.side ^ 2 = 185 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_line_area_of_square_ABCD_l1115_111511


namespace NUMINAMATH_CALUDE_transformed_roots_l1115_111558

-- Define the original quadratic equation
def original_quadratic (p q r x : ℝ) : Prop := p * x^2 + q * x + r = 0

-- Define the roots of the original quadratic equation
def has_roots (p q r u v : ℝ) : Prop := original_quadratic p q r u ∧ original_quadratic p q r v

-- Define the new quadratic equation
def new_quadratic (p q r x : ℝ) : Prop := x^2 - 4 * q * x + 4 * p * r + 3 * q^2 = 0

-- Theorem statement
theorem transformed_roots (p q r u v : ℝ) (hp : p ≠ 0) :
  has_roots p q r u v →
  new_quadratic p q r (2 * p * u + 3 * q) ∧ new_quadratic p q r (2 * p * v + 3 * q) :=
by sorry

end NUMINAMATH_CALUDE_transformed_roots_l1115_111558


namespace NUMINAMATH_CALUDE_line_through_points_l1115_111523

def point_on_line (x y x1 y1 x2 y2 : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

theorem line_through_points : 
  point_on_line 4 8 8 16 2 4 ∧
  point_on_line 6 12 8 16 2 4 ∧
  point_on_line 10 20 8 16 2 4 ∧
  ¬ point_on_line 5 11 8 16 2 4 ∧
  ¬ point_on_line 3 7 8 16 2 4 :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_l1115_111523


namespace NUMINAMATH_CALUDE_fair_coin_probability_l1115_111570

def n : ℕ := 5
def k : ℕ := 2
def p : ℚ := 1/2

theorem fair_coin_probability : 
  (n.choose k) * p^k * (1 - p)^(n - k) = 10/32 := by sorry

end NUMINAMATH_CALUDE_fair_coin_probability_l1115_111570
