import Mathlib

namespace NUMINAMATH_CALUDE_rational_irrational_relations_l2098_209821

theorem rational_irrational_relations (m n : ℚ) :
  (((m - 3) * Real.sqrt 6 + n - 3 = 0) → Real.sqrt (m * n) = 3 ∨ Real.sqrt (m * n) = -3) ∧
  ((∃ x : ℝ, m^2 = x ∧ n^2 = x ∧ (2 + Real.sqrt 3) * m - (1 - Real.sqrt 3) * n = 5) → 
   ∃ x : ℝ, m^2 = x ∧ n^2 = x ∧ x = 25/9) :=
by sorry

end NUMINAMATH_CALUDE_rational_irrational_relations_l2098_209821


namespace NUMINAMATH_CALUDE_polynomial_roots_l2098_209894

theorem polynomial_roots : ∃ (x₁ x₂ x₃ x₄ : ℝ), 
  (x₁ = 0 ∧ x₂ = 1/3 ∧ x₃ = 2 ∧ x₄ = -5) ∧
  (∀ x : ℝ, 3*x^4 + 11*x^3 - 28*x^2 + 10*x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2098_209894


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2098_209898

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  x + 2*y ≥ 9 ∧ ∀ M : ℝ, ∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧ 1/x' + 2/y' = 1 ∧ x' + 2*y' > M :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2098_209898


namespace NUMINAMATH_CALUDE_area_of_triangle_APB_l2098_209860

-- Define the square and point P
def Square (s : ℝ) := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ s ∧ 0 ≤ p.2 ∧ p.2 ≤ s}

def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 0)
def C : ℝ × ℝ := (8, 8)
def D : ℝ × ℝ := (0, 8)
def F : ℝ × ℝ := (4, 8)

-- Define the conditions
def PointInSquare (P : ℝ × ℝ) : Prop := P ∈ Square 8

def EqualSegments (P : ℝ × ℝ) : Prop :=
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2

def PerpendicularPC_FD (P : ℝ × ℝ) : Prop :=
  (P.1 - C.1) * (F.1 - D.1) + (P.2 - C.2) * (F.2 - D.2) = 0

def PerpendicularPB_AC (P : ℝ × ℝ) : Prop :=
  (P.1 - B.1) * (A.1 - C.1) + (P.2 - B.2) * (A.2 - C.2) = 0

-- Theorem statement
theorem area_of_triangle_APB (P : ℝ × ℝ) 
  (h1 : PointInSquare P) 
  (h2 : EqualSegments P) 
  (h3 : PerpendicularPC_FD P) 
  (h4 : PerpendicularPB_AC P) : 
  ∃ (area : ℝ), area = 32/5 ∧ 
  area = (1/2) * ((P.1 - A.1)^2 + (P.2 - A.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_APB_l2098_209860


namespace NUMINAMATH_CALUDE_inequality_proof_l2098_209843

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2098_209843


namespace NUMINAMATH_CALUDE_rectangular_sheet_area_l2098_209816

theorem rectangular_sheet_area :
  ∀ (area_small area_large total_area : ℝ),
  area_large = 4 * area_small →
  area_large - area_small = 2208 →
  total_area = area_small + area_large →
  total_area = 3680 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_sheet_area_l2098_209816


namespace NUMINAMATH_CALUDE_congruent_side_length_for_specific_triangle_l2098_209892

/-- Represents an isosceles triangle with base length and area -/
structure IsoscelesTriangle where
  base : ℝ
  area : ℝ

/-- Calculates the length of a congruent side in an isosceles triangle -/
def congruentSideLength (triangle : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem stating that for an isosceles triangle with base 30 and area 72, 
    the length of a congruent side is 15.75 -/
theorem congruent_side_length_for_specific_triangle :
  let triangle : IsoscelesTriangle := { base := 30, area := 72 }
  congruentSideLength triangle = 15.75 := by sorry

end NUMINAMATH_CALUDE_congruent_side_length_for_specific_triangle_l2098_209892


namespace NUMINAMATH_CALUDE_function_with_two_symmetry_centers_decomposition_l2098_209881

/-- A function has a center of symmetry at a if f(a-x) + f(a+x) = 2f(a) for all real x -/
def HasCenterOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a - x) + f (a + x) = 2 * f a

/-- A function is linear if f(x) = mx + b for some real m and b -/
def IsLinear (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

/-- A function is periodic if there exists a non-zero real number p such that f(x + p) = f(x) for all real x -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x : ℝ, f (x + p) = f x

/-- Main theorem: A function with at least two centers of symmetry can be written as the sum of a linear function and a periodic function -/
theorem function_with_two_symmetry_centers_decomposition (f : ℝ → ℝ) 
  (h1 : ∃ p q : ℝ, p ≠ q ∧ HasCenterOfSymmetry f p ∧ HasCenterOfSymmetry f q) :
  ∃ g h : ℝ → ℝ, IsLinear g ∧ IsPeriodic h ∧ ∀ x : ℝ, f x = g x + h x := by
  sorry


end NUMINAMATH_CALUDE_function_with_two_symmetry_centers_decomposition_l2098_209881


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2098_209857

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (30 : ℝ) / 360 * (2 * Real.pi * r₁) = (24 : ℝ) / 360 * (2 * Real.pi * r₂) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2098_209857


namespace NUMINAMATH_CALUDE_second_player_wins_l2098_209889

/-- Represents a grid in the domino gluing game -/
structure Grid :=
  (size : Nat)
  (is_cut_into_dominoes : Bool)

/-- Represents a move in the domino gluing game -/
structure Move :=
  (x1 y1 x2 y2 : Nat)

/-- Represents the state of the game -/
structure GameState :=
  (grid : Grid)
  (current_player : Nat)
  (moves : List Move)

/-- Checks if a move is valid -/
def is_valid_move (state : GameState) (move : Move) : Bool :=
  sorry

/-- Checks if the game is over (i.e., the figure is connected) -/
def is_game_over (state : GameState) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- Checks if a strategy is winning for a player -/
def is_winning_strategy (player : Nat) (strategy : Strategy) : Prop :=
  sorry

/-- The main theorem: the second player has a winning strategy -/
theorem second_player_wins (grid : Grid) 
    (h1 : grid.size = 100) 
    (h2 : grid.is_cut_into_dominoes = true) : 
  ∃ (strategy : Strategy), is_winning_strategy 2 strategy :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l2098_209889


namespace NUMINAMATH_CALUDE_unique_solution_to_x_equals_negative_x_l2098_209861

theorem unique_solution_to_x_equals_negative_x : 
  ∀ x : ℝ, x = -x ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_solution_to_x_equals_negative_x_l2098_209861


namespace NUMINAMATH_CALUDE_lines_in_same_plane_l2098_209866

-- Define the necessary types
variable (Point Line Plane : Type)

-- Define the necessary relations
variable (lies_in : Point → Line → Prop)
variable (lies_in_plane : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem lines_in_same_plane 
  (a b : Line) 
  (α : Plane) 
  (h1 : intersect a b) 
  (h2 : lies_in_plane a α) 
  (h3 : lies_in_plane b α) :
  ∀ (c : Line), parallel c b → (∃ (p : Point), lies_in p a ∧ lies_in p c) → 
  lies_in_plane c α :=
sorry

end NUMINAMATH_CALUDE_lines_in_same_plane_l2098_209866


namespace NUMINAMATH_CALUDE_square_area_difference_l2098_209830

/-- Given two line segments where one is 2 cm longer than the other, and the difference 
    of the areas of squares drawn on these line segments is 32 sq. cm, 
    prove that the length of the longer line segment is 9 cm. -/
theorem square_area_difference (x : ℝ) 
  (h1 : (x + 2)^2 - x^2 = 32) : 
  x + 2 = 9 := by sorry

end NUMINAMATH_CALUDE_square_area_difference_l2098_209830


namespace NUMINAMATH_CALUDE_boy_age_problem_l2098_209803

theorem boy_age_problem (current_age : ℕ) (years_ago : ℕ) : 
  current_age = 10 →
  current_age = 2 * (current_age - years_ago) →
  years_ago = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_boy_age_problem_l2098_209803


namespace NUMINAMATH_CALUDE_randy_biscuits_left_l2098_209897

/-- The number of biscuits Randy is left with after receiving and losing some -/
def biscuits_left (initial : ℕ) (from_father : ℕ) (from_mother : ℕ) (eaten_by_brother : ℕ) : ℕ :=
  initial + from_father + from_mother - eaten_by_brother

/-- Theorem stating that Randy is left with 40 biscuits -/
theorem randy_biscuits_left : biscuits_left 32 13 15 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_randy_biscuits_left_l2098_209897


namespace NUMINAMATH_CALUDE_total_days_is_30_l2098_209852

/-- The number of days being considered -/
def total_days : ℕ := sorry

/-- The mean daily profit for all days in rupees -/
def mean_profit : ℕ := 350

/-- The mean profit for the first 15 days in rupees -/
def mean_profit_first_15 : ℕ := 275

/-- The mean profit for the last 15 days in rupees -/
def mean_profit_last_15 : ℕ := 425

/-- Theorem stating that the total number of days is 30 -/
theorem total_days_is_30 :
  total_days = 30 ∧
  total_days * mean_profit = 15 * mean_profit_first_15 + 15 * mean_profit_last_15 :=
by sorry

end NUMINAMATH_CALUDE_total_days_is_30_l2098_209852


namespace NUMINAMATH_CALUDE_problem_statement_l2098_209859

theorem problem_statement (x y z : ℝ) (hx : x = 2) (hy : y = -3) (hz : z = 1) :
  x^2 + y^2 + z^2 + 2*x*y - z^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2098_209859


namespace NUMINAMATH_CALUDE_factorization_theorem_l2098_209834

theorem factorization_theorem (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2*y) * (x + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l2098_209834


namespace NUMINAMATH_CALUDE_checkerboard_theorem_l2098_209836

def board_size : Nat := 9
def num_lines : Nat := 10

/-- The number of rectangles on the checkerboard -/
def num_rectangles : Nat := (num_lines.choose 2) * (num_lines.choose 2)

/-- The number of squares on the checkerboard -/
def num_squares : Nat := (board_size * (board_size + 1) * (2 * board_size + 1)) / 6

/-- The ratio of squares to rectangles -/
def ratio : Rat := num_squares / num_rectangles

theorem checkerboard_theorem :
  num_rectangles = 2025 ∧
  num_squares = 285 ∧
  ratio = 19 / 135 ∧
  19 + 135 = 154 := by sorry

end NUMINAMATH_CALUDE_checkerboard_theorem_l2098_209836


namespace NUMINAMATH_CALUDE_debate_students_difference_l2098_209877

theorem debate_students_difference (s1 s2 s3 : ℕ) : 
  s1 = 2 * s2 →
  s3 = 200 →
  s1 + s2 + s3 = 920 →
  s2 - s3 = 40 := by
sorry

end NUMINAMATH_CALUDE_debate_students_difference_l2098_209877


namespace NUMINAMATH_CALUDE_unique_prime_divides_sigma_pred_l2098_209891

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- Theorem: The only prime p that divides σ(p-1) is 3 -/
theorem unique_prime_divides_sigma_pred :
  ∀ p : ℕ, Nat.Prime p → (p ∣ sigma (p - 1) ↔ p = 3) := by sorry

end NUMINAMATH_CALUDE_unique_prime_divides_sigma_pred_l2098_209891


namespace NUMINAMATH_CALUDE_ball_selection_count_l2098_209806

/-- Represents the number of balls of each color -/
def ballsPerColor : ℕ := 7

/-- Represents the number of colors -/
def numberOfColors : ℕ := 3

/-- Represents the total number of balls -/
def totalBalls : ℕ := ballsPerColor * numberOfColors

/-- Checks if three numbers are non-consecutive -/
def areNonConsecutive (a b c : ℕ) : Prop :=
  (a + 1 ≠ b ∧ b + 1 ≠ c) ∧ (b + 1 ≠ a ∧ c + 1 ≠ b) ∧ (c + 1 ≠ a ∧ a + 1 ≠ c)

/-- Counts the number of ways to select 3 non-consecutive numbers from 1 to 7 -/
def nonConsecutiveSelections : ℕ := 35

/-- The main theorem to be proved -/
theorem ball_selection_count :
  (∃ (f : Fin totalBalls → ℕ × Fin numberOfColors),
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∀ i, (f i).1 ∈ Finset.range ballsPerColor) ∧
    (∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      areNonConsecutive (f a).1 (f b).1 (f c).1 ∧
      (f a).2 ≠ (f b).2 ∧ (f b).2 ≠ (f c).2 ∧ (f a).2 ≠ (f c).2)) →
  nonConsecutiveSelections * numberOfColors = 60 :=
sorry

end NUMINAMATH_CALUDE_ball_selection_count_l2098_209806


namespace NUMINAMATH_CALUDE_sin_plus_two_cos_for_point_l2098_209838

/-- Given a point P(-3,4) on the terminal side of angle α, prove that sin α + 2 cos α = -2/5 -/
theorem sin_plus_two_cos_for_point (α : Real) :
  let P : ℝ × ℝ := (-3, 4)
  let r : ℝ := Real.sqrt (P.1^2 + P.2^2)
  Real.sin α = P.2 / r ∧ Real.cos α = P.1 / r →
  Real.sin α + 2 * Real.cos α = -2/5 := by
sorry


end NUMINAMATH_CALUDE_sin_plus_two_cos_for_point_l2098_209838


namespace NUMINAMATH_CALUDE_sum_of_digits_base_8_of_888_l2098_209848

/-- The sum of the digits of the base 8 representation of 888₁₀ is 13. -/
theorem sum_of_digits_base_8_of_888 : 
  (Nat.digits 8 888).sum = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_base_8_of_888_l2098_209848


namespace NUMINAMATH_CALUDE_existence_of_polynomial_and_c1_value_l2098_209862

/-- D(m) counts the number of quadruples (a₁, a₂, a₃, a₄) of distinct integers 
    with 1 ≤ aᵢ ≤ m for all i such that m divides a₁+a₂+a₃+a₄ -/
def D (m : ℕ) : ℕ := sorry

/-- The polynomial q(x) = c₃x³ + c₂x² + c₁x + c₀ -/
def q (x : ℕ) : ℕ := sorry

theorem existence_of_polynomial_and_c1_value :
  ∃ (c₃ c₂ c₁ c₀ : ℤ), 
    (∀ m : ℕ, m ≥ 5 → Odd m → D m = c₃ * m^3 + c₂ * m^2 + c₁ * m + c₀) ∧ 
    c₁ = 11 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_polynomial_and_c1_value_l2098_209862


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2098_209872

theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, 4 * x^2 - (4*k - 2) * x + k^2 = 0) → k ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2098_209872


namespace NUMINAMATH_CALUDE_ara_current_height_l2098_209839

/-- Represents a person's height and growth --/
structure Person where
  originalHeight : ℝ
  growthFactor : ℝ

/-- Calculates the current height of a person given their original height and growth factor --/
def currentHeight (p : Person) : ℝ := p.originalHeight * (1 + p.growthFactor)

/-- Theorem stating Ara's current height given the conditions --/
theorem ara_current_height (shea ara : Person) 
  (h1 : shea.growthFactor = 0.25)
  (h2 : currentHeight shea = 75)
  (h3 : ara.originalHeight = shea.originalHeight)
  (h4 : ara.growthFactor = shea.growthFactor / 3) :
  currentHeight ara = 65 := by
  sorry


end NUMINAMATH_CALUDE_ara_current_height_l2098_209839


namespace NUMINAMATH_CALUDE_floor_ceiling_expression_l2098_209887

theorem floor_ceiling_expression : 
  ⌊⌈(12 / 5 : ℚ)^2⌉ * 3 + 14 / 3⌋ = 22 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_expression_l2098_209887


namespace NUMINAMATH_CALUDE_origin_outside_circle_l2098_209850

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) :
  let circle := fun (x y : ℝ) => x^2 + y^2 + 2*a*x + 2*y + (a-1)^2 = 0
  ¬ circle 0 0 := by
  sorry

end NUMINAMATH_CALUDE_origin_outside_circle_l2098_209850


namespace NUMINAMATH_CALUDE_pursuer_catches_pursued_l2098_209899

/-- Represents a point on an infinite straight line -/
structure Point where
  position : ℝ

/-- Represents a moving object on the line -/
structure MovingObject where
  initialPosition : Point
  speed : ℝ
  direction : Bool  -- True for positive direction, False for negative

/-- The pursuer (new police car) -/
def pursuer : MovingObject := {
  initialPosition := { position := 0 },
  speed := 1,  -- Normalized to 1
  direction := true  -- Arbitrary initial direction
}

/-- The pursued (stolen police car) -/
def pursued : MovingObject := {
  initialPosition := { position := 0 },  -- Arbitrary initial position
  speed := 0.9,  -- 90% of pursuer's speed
  direction := true  -- Arbitrary initial direction
}

/-- Theorem stating that the pursuer can always catch the pursued -/
theorem pursuer_catches_pursued :
  ∃ (t : ℝ), t ≥ 0 ∧ 
  pursuer.initialPosition.position + t * pursuer.speed = 
  pursued.initialPosition.position + t * pursued.speed :=
sorry

end NUMINAMATH_CALUDE_pursuer_catches_pursued_l2098_209899


namespace NUMINAMATH_CALUDE_infinite_nonprime_powers_l2098_209868

theorem infinite_nonprime_powers (k : ℕ) : ∃ n : ℕ, n ≥ k ∧
  (¬ Nat.Prime (2^(2^n) + 1) ∨ ¬ Nat.Prime (2018^(2^n) + 1)) := by
  sorry

end NUMINAMATH_CALUDE_infinite_nonprime_powers_l2098_209868


namespace NUMINAMATH_CALUDE_macy_running_goal_l2098_209818

/-- Calculates the remaining miles to reach a weekly running goal -/
def remaining_miles (weekly_goal : ℕ) (daily_miles : ℕ) (days_run : ℕ) : ℕ :=
  weekly_goal - (daily_miles * days_run)

/-- Proves that given a weekly goal of 24 miles, running 3 miles per day for 6 days,
    the remaining miles to reach the goal is 6 miles -/
theorem macy_running_goal :
  remaining_miles 24 3 6 = 6 := by
sorry

end NUMINAMATH_CALUDE_macy_running_goal_l2098_209818


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2098_209828

theorem complex_equation_solution (a : ℝ) (i : ℂ) : 
  i * i = -1 → (a - i)^2 = 2*i → a = -1 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2098_209828


namespace NUMINAMATH_CALUDE_sufficient_condition_absolute_value_l2098_209832

theorem sufficient_condition_absolute_value (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_absolute_value_l2098_209832


namespace NUMINAMATH_CALUDE_two_fifths_of_number_l2098_209886

theorem two_fifths_of_number (n : ℝ) : (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * n = 16 → (2/5 : ℝ) * n = 192 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_of_number_l2098_209886


namespace NUMINAMATH_CALUDE_ritas_money_theorem_l2098_209826

/-- Calculates the remaining money after Rita's purchases --/
def ritas_remaining_money (initial_amount dresses_cost pants_cost jackets_cost transportation : ℕ) : ℕ :=
  initial_amount - (5 * dresses_cost + 3 * pants_cost + 4 * jackets_cost + transportation)

/-- Theorem stating that Rita's remaining money is 139 --/
theorem ritas_money_theorem :
  ritas_remaining_money 400 20 12 30 5 = 139 := by
  sorry

end NUMINAMATH_CALUDE_ritas_money_theorem_l2098_209826


namespace NUMINAMATH_CALUDE_pyramid_sum_is_25_l2098_209845

/-- Calculates the sum of blocks in a pyramid with given parameters -/
def pyramidSum (levels : Nat) (firstRowBlocks : Nat) (decrease : Nat) : Nat :=
  let blockSequence := List.range levels |>.map (fun i => firstRowBlocks - i * decrease)
  blockSequence.sum

/-- The sum of blocks in a 5-level pyramid with specific parameters is 25 -/
theorem pyramid_sum_is_25 : pyramidSum 5 9 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_sum_is_25_l2098_209845


namespace NUMINAMATH_CALUDE_dice_sum_not_sixteen_l2098_209874

theorem dice_sum_not_sixteen (a b c d e : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  1 ≤ e ∧ e ≤ 6 →
  a * b * c * d * e = 72 →
  a + b + c + d + e ≠ 16 := by
sorry

end NUMINAMATH_CALUDE_dice_sum_not_sixteen_l2098_209874


namespace NUMINAMATH_CALUDE_product_of_terms_l2098_209819

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = r * b n

/-- The main theorem -/
theorem product_of_terms (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  3 * a 1 - (a 8)^2 + 3 * a 15 = 0 →
  a 8 = b 10 →
  b 3 * b 17 = 36 := by
  sorry

end NUMINAMATH_CALUDE_product_of_terms_l2098_209819


namespace NUMINAMATH_CALUDE_smallest_n_for_cube_T_l2098_209867

/-- Function that calculates (n+2)3^n for a positive integer n -/
def T (n : ℕ+) : ℕ := (n + 2) * 3^(n : ℕ)

/-- Predicate to check if a natural number is a perfect cube -/
def is_cube (m : ℕ) : Prop := ∃ k : ℕ, m = k^3

/-- Theorem stating that 1 is the smallest positive integer n for which T(n) is a perfect cube -/
theorem smallest_n_for_cube_T :
  (∃ n : ℕ+, is_cube (T n)) ∧ (∀ n : ℕ+, is_cube (T n) → 1 ≤ n) := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_cube_T_l2098_209867


namespace NUMINAMATH_CALUDE_model_a_sample_size_l2098_209856

/-- Calculates the number of items to select in stratified sampling -/
def stratified_sample_size (total_production : ℕ) (model_production : ℕ) (sample_size : ℕ) : ℕ :=
  (model_production * sample_size) / total_production

/-- Proves that the stratified sample size for Model A is 6 -/
theorem model_a_sample_size :
  stratified_sample_size 9200 1200 46 = 6 := by
sorry

end NUMINAMATH_CALUDE_model_a_sample_size_l2098_209856


namespace NUMINAMATH_CALUDE_time_after_2021_hours_l2098_209804

/-- Represents a day of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a time of day -/
structure TimeOfDay where
  hour : Nat
  minute : Nat
  h_valid : hour < 24
  m_valid : minute < 60

/-- Represents a moment in time -/
structure Moment where
  day : DayOfWeek
  time : TimeOfDay

/-- Adds hours to a given moment and returns the new moment -/
def addHours (start : Moment) (hours : Nat) : Moment :=
  sorry

theorem time_after_2021_hours :
  let start : Moment := ⟨DayOfWeek.Monday, ⟨20, 21, sorry, sorry⟩⟩
  let end_moment : Moment := addHours start 2021
  end_moment = ⟨DayOfWeek.Tuesday, ⟨1, 21, sorry, sorry⟩⟩ := by
  sorry

end NUMINAMATH_CALUDE_time_after_2021_hours_l2098_209804


namespace NUMINAMATH_CALUDE_prob_at_least_three_even_is_five_sixteenths_l2098_209875

/-- Probability of rolling an even number on a fair die -/
def prob_even : ℚ := 1/2

/-- Number of rolls -/
def num_rolls : ℕ := 4

/-- Probability of rolling an even number at least three times in four rolls -/
def prob_at_least_three_even : ℚ :=
  Nat.choose num_rolls 3 * prob_even^3 * (1 - prob_even) +
  Nat.choose num_rolls 4 * prob_even^4

theorem prob_at_least_three_even_is_five_sixteenths :
  prob_at_least_three_even = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_three_even_is_five_sixteenths_l2098_209875


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2098_209863

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → c = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2098_209863


namespace NUMINAMATH_CALUDE_integer_chord_lines_count_l2098_209805

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  point : Point
  direction : Point  -- Direction vector

/-- Define the circle from the problem -/
def problemCircle : Circle :=
  { center := { x := 2, y := -2 },
    radius := 5 }

/-- Define the point M -/
def pointM : Point :=
  { x := 2, y := 2 }

/-- Function to check if a point is inside a circle -/
def isInside (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

/-- Function to count lines passing through M that cut off integer-length chords -/
def countIntegerChordLines (c : Circle) (m : Point) : ℕ :=
  sorry  -- Implementation details omitted

/-- The main theorem -/
theorem integer_chord_lines_count :
  isInside pointM problemCircle →
  countIntegerChordLines problemCircle pointM = 8 := by
  sorry

end NUMINAMATH_CALUDE_integer_chord_lines_count_l2098_209805


namespace NUMINAMATH_CALUDE_milton_pies_sold_l2098_209888

/-- Calculates the total number of pies sold given the number of slices ordered and slices per pie -/
def total_pies_sold (apple_slices_ordered : ℕ) (peach_slices_ordered : ℕ) 
                    (slices_per_apple_pie : ℕ) (slices_per_peach_pie : ℕ) : ℕ :=
  (apple_slices_ordered / slices_per_apple_pie) + (peach_slices_ordered / slices_per_peach_pie)

/-- Theorem stating that given the specific conditions, Milton sold 15 pies -/
theorem milton_pies_sold : 
  total_pies_sold 56 48 8 6 = 15 := by
  sorry

#eval total_pies_sold 56 48 8 6

end NUMINAMATH_CALUDE_milton_pies_sold_l2098_209888


namespace NUMINAMATH_CALUDE_mean_motorcycles_rainy_days_l2098_209800

def sunny_car_counts : List ℝ := [30, 14, 14, 21, 25]
def sunny_motorcycle_counts : List ℝ := [5, 2, 4, 1, 3]
def rainy_car_counts : List ℝ := [40, 20, 17, 31, 30]
def rainy_motorcycle_counts : List ℝ := [2, 1, 1, 0, 2]

theorem mean_motorcycles_rainy_days :
  (rainy_motorcycle_counts.sum / rainy_motorcycle_counts.length : ℝ) = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_mean_motorcycles_rainy_days_l2098_209800


namespace NUMINAMATH_CALUDE_oleg_event_guests_l2098_209890

theorem oleg_event_guests (total_guests men : ℕ) (h1 : total_guests = 80) (h2 : men = 40) :
  let women := men / 2
  let adults := men + women
  let original_children := total_guests - adults
  original_children + 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_oleg_event_guests_l2098_209890


namespace NUMINAMATH_CALUDE_no_arithmetic_progression_l2098_209855

theorem no_arithmetic_progression : 
  ¬∃ (y : ℝ), (∃ (d : ℝ), (3*y + 1) - (y - 3) = d ∧ (5*y - 7) - (3*y + 1) = d) := by
  sorry

end NUMINAMATH_CALUDE_no_arithmetic_progression_l2098_209855


namespace NUMINAMATH_CALUDE_cubic_polynomial_sum_l2098_209831

/-- Given a cubic polynomial Q with specific values at 1, -1, and 0,
    prove that Q(3) + Q(-3) = 47m -/
theorem cubic_polynomial_sum (m : ℝ) (Q : ℝ → ℝ) 
  (h_cubic : ∃ (a b c : ℝ), ∀ x, Q x = a * x^3 + b * x^2 + c * x + m)
  (h_1 : Q 1 = 3 * m)
  (h_neg1 : Q (-1) = 4 * m)
  (h_0 : Q 0 = m) :
  Q 3 + Q (-3) = 47 * m := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_sum_l2098_209831


namespace NUMINAMATH_CALUDE_total_matting_cost_l2098_209813

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a room with its dimensions and matting cost -/
structure Room where
  dimensions : RoomDimensions
  mattingCostPerSquareMeter : ℝ

/-- Calculates the floor area of a room -/
def floorArea (room : Room) : ℝ :=
  room.dimensions.length * room.dimensions.width

/-- Calculates the matting cost for a room -/
def mattingCost (room : Room) : ℝ :=
  floorArea room * room.mattingCostPerSquareMeter

/-- The three rooms in the house -/
def hall : Room :=
  { dimensions := { length := 20, width := 15, height := 5 },
    mattingCostPerSquareMeter := 40 }

def bedroom : Room :=
  { dimensions := { length := 10, width := 5, height := 4 },
    mattingCostPerSquareMeter := 35 }

def study : Room :=
  { dimensions := { length := 8, width := 6, height := 3 },
    mattingCostPerSquareMeter := 45 }

/-- Theorem: The total cost of matting for all three rooms is 15910 -/
theorem total_matting_cost :
  mattingCost hall + mattingCost bedroom + mattingCost study = 15910 := by
  sorry

end NUMINAMATH_CALUDE_total_matting_cost_l2098_209813


namespace NUMINAMATH_CALUDE_radio_survey_female_nonlisteners_l2098_209878

theorem radio_survey_female_nonlisteners (total_surveyed : ℕ) 
  (males_listen females_dont_listen total_listen total_dont_listen : ℕ) :
  total_surveyed = total_listen + total_dont_listen →
  males_listen ≤ total_listen →
  females_dont_listen ≤ total_dont_listen →
  total_surveyed = 255 →
  males_listen = 45 →
  total_listen = 120 →
  total_dont_listen = 135 →
  females_dont_listen = 87 →
  females_dont_listen = 87 :=
by sorry

end NUMINAMATH_CALUDE_radio_survey_female_nonlisteners_l2098_209878


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_greater_than_one_l2098_209817

theorem sum_of_reciprocals_greater_than_one 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ > 1) 
  (h₂ : a₂ > 1) 
  (h₃ : a₃ > 1) 
  (hS : a₁ + a₂ + a₃ = a₁ + a₂ + a₃) 
  (hcond₁ : a₁^2 / (a₁ - 1) > a₁ + a₂ + a₃) 
  (hcond₂ : a₂^2 / (a₂ - 1) > a₁ + a₂ + a₃) 
  (hcond₃ : a₃^2 / (a₃ - 1) > a₁ + a₂ + a₃) : 
  1 / (a₁ + a₂) + 1 / (a₂ + a₃) + 1 / (a₃ + a₁) > 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_greater_than_one_l2098_209817


namespace NUMINAMATH_CALUDE_two_digit_perfect_square_divisible_by_five_l2098_209847

theorem two_digit_perfect_square_divisible_by_five :
  ∃! n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ ∃ m : ℕ, n = m^2 ∧ n % 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_perfect_square_divisible_by_five_l2098_209847


namespace NUMINAMATH_CALUDE_percentage_relation_l2098_209833

theorem percentage_relation (x a b : ℝ) (h1 : a = 0.05 * x) (h2 : b = 0.25 * x) :
  a = 0.2 * b := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l2098_209833


namespace NUMINAMATH_CALUDE_equation_solution_l2098_209880

theorem equation_solution (x : ℝ) : (4 + 2*x) / (7 + x) = (2 + x) / (3 + x) ↔ x = -2 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2098_209880


namespace NUMINAMATH_CALUDE_second_group_cost_l2098_209854

/-- The cost of a hotdog in dollars -/
def hotdog_cost : ℚ := 1/2

/-- The cost of a soft drink in dollars -/
def soft_drink_cost : ℚ := 1/2

/-- The number of hotdogs purchased by the first group -/
def first_group_hotdogs : ℕ := 10

/-- The number of soft drinks purchased by the first group -/
def first_group_drinks : ℕ := 5

/-- The total cost of the first group's purchase in dollars -/
def first_group_total : ℚ := 25/2

/-- The number of hotdogs purchased by the second group -/
def second_group_hotdogs : ℕ := 7

/-- The number of soft drinks purchased by the second group -/
def second_group_drinks : ℕ := 4

/-- Theorem stating that the cost of the second group's purchase is $5.50 -/
theorem second_group_cost : 
  (second_group_hotdogs : ℚ) * hotdog_cost + (second_group_drinks : ℚ) * soft_drink_cost = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_second_group_cost_l2098_209854


namespace NUMINAMATH_CALUDE_factorial_sum_remainder_l2098_209869

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem factorial_sum_remainder (n : ℕ) : 
  n ≥ 50 → sum_factorials n % 25 = (factorial 1 + factorial 2 + factorial 3 + factorial 4) % 25 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_remainder_l2098_209869


namespace NUMINAMATH_CALUDE_marked_squares_theorem_l2098_209835

/-- A type representing a table with marked squares -/
def MarkedTable (n : ℕ) := Fin n → Fin n → Bool

/-- A function that checks if a square is on or above the main diagonal -/
def isAboveDiagonal {n : ℕ} (i j : Fin n) : Bool :=
  i.val ≤ j.val

/-- A function that counts the number of marked squares in a table -/
def countMarkedSquares {n : ℕ} (table : MarkedTable n) : ℕ :=
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => if table i j then 1 else 0))

/-- A predicate that checks if a table can be rearranged to satisfy the condition -/
def canRearrange {n : ℕ} (table : MarkedTable n) : Prop :=
  ∃ (rowPerm colPerm : Equiv.Perm (Fin n)),
    ∀ i j, table i j → isAboveDiagonal (rowPerm i) (colPerm j)

theorem marked_squares_theorem (n : ℕ) (h : n > 1) :
  ∀ (table : MarkedTable n),
    canRearrange table ↔ countMarkedSquares table ≤ n + 1 :=
by sorry

end NUMINAMATH_CALUDE_marked_squares_theorem_l2098_209835


namespace NUMINAMATH_CALUDE_problem_solution_l2098_209858

theorem problem_solution (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + 2 / b = d) (h2 : b + 2 / c = d) (h3 : c + 2 / a = d) :
  d = Real.sqrt 2 ∨ d = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2098_209858


namespace NUMINAMATH_CALUDE_flower_bed_area_and_perimeter_l2098_209820

/-- Represents a rectangular flower bed -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Calculate the area of a rectangular flower bed -/
def area (fb : FlowerBed) : ℝ :=
  fb.length * fb.width

/-- Calculate the perimeter of a rectangular flower bed -/
def perimeter (fb : FlowerBed) : ℝ :=
  2 * (fb.length + fb.width)

theorem flower_bed_area_and_perimeter :
  let fb : FlowerBed := { length := 60, width := 45 }
  area fb = 2700 ∧ perimeter fb = 210 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_area_and_perimeter_l2098_209820


namespace NUMINAMATH_CALUDE_no_roots_around_1000_l2098_209864

/-- A quadratic trinomial -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic trinomial at a given x -/
def QuadraticTrinomial.value (q : QuadraticTrinomial) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- A list of quadratic trinomials -/
def trinomials : List QuadraticTrinomial := sorry

/-- The sum of all quadratic trinomials in the list -/
def f (x : ℝ) : ℝ :=
  (trinomials.map (fun q => q.value x)).sum

/-- All trinomials are positive at x = 1000 -/
axiom all_positive : ∀ q ∈ trinomials, q.value 1000 > 0

/-- Theorem: It's impossible for f to have one root less than 1000 and another greater than 1000 -/
theorem no_roots_around_1000 : ¬∃ (r₁ r₂ : ℝ), r₁ < 1000 ∧ r₂ > 1000 ∧ f r₁ = 0 ∧ f r₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_roots_around_1000_l2098_209864


namespace NUMINAMATH_CALUDE_otimes_four_otimes_four_four_l2098_209896

-- Define the binary operation ⊗
def otimes (x y : ℝ) : ℝ := x^3 + 3*x*y - y

-- Theorem statement
theorem otimes_four_otimes_four_four : otimes 4 (otimes 4 4) = 1252 := by
  sorry

end NUMINAMATH_CALUDE_otimes_four_otimes_four_four_l2098_209896


namespace NUMINAMATH_CALUDE_rounding_317500_equals_31_8_ten_thousand_l2098_209883

-- Define rounding to the nearest thousand
def roundToThousand (n : ℕ) : ℕ :=
  (n + 500) / 1000 * 1000

-- Define representation in ten thousands
def toTenThousand (n : ℕ) : ℚ :=
  n / 10000

-- Theorem statement
theorem rounding_317500_equals_31_8_ten_thousand :
  toTenThousand (roundToThousand 317500) = 31.8 := by
  sorry

end NUMINAMATH_CALUDE_rounding_317500_equals_31_8_ten_thousand_l2098_209883


namespace NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_two_l2098_209825

theorem no_solution_iff_m_eq_neg_two (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → (x - 5) / (x - 3) ≠ m / (x - 3) + 2) ↔ m = -2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_two_l2098_209825


namespace NUMINAMATH_CALUDE_complex_expression_equals_nine_l2098_209882

theorem complex_expression_equals_nine :
  (Real.rpow 1.5 (1/3) * Real.rpow 12 (1/6))^2 + 8 * Real.rpow 1 0.75 - Real.rpow (-1/4) (-2) - 5 * Real.rpow 0.125 0 = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_nine_l2098_209882


namespace NUMINAMATH_CALUDE_ages_relationship_l2098_209807

/-- Given the ages of Katherine (K), Mel (M), and Lexi (L), with the relationships
    M = K - 3 and L = M + 2, prove that when K = 60, M = 57 and L = 59. -/
theorem ages_relationship (K M L : ℕ) 
    (h1 : M = K - 3) 
    (h2 : L = M + 2) 
    (h3 : K = 60) : 
  M = 57 ∧ L = 59 := by
sorry

end NUMINAMATH_CALUDE_ages_relationship_l2098_209807


namespace NUMINAMATH_CALUDE_graph_translation_l2098_209801

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define k as a positive real number
variable (k : ℝ)
variable (h : k > 0)

-- State the theorem
theorem graph_translation (x y : ℝ) : 
  y = f (x + k) ↔ y = f ((x + k) - k) :=
sorry

end NUMINAMATH_CALUDE_graph_translation_l2098_209801


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l2098_209865

theorem point_in_third_quadrant (a b : ℝ) (h : a < b ∧ b < 0) :
  (a - b < 0) ∧ (b < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l2098_209865


namespace NUMINAMATH_CALUDE_chess_tournament_matches_l2098_209802

/-- The number of matches in a round-robin tournament with n players -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament_matches :
  num_matches 10 = 45 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_matches_l2098_209802


namespace NUMINAMATH_CALUDE_interest_rate_equation_l2098_209837

/-- Given a principal that doubles in 10 years with semiannual compounding,
    this theorem states the equation that the annual interest rate must satisfy. -/
theorem interest_rate_equation (r : ℝ) : 
  (∀ P : ℝ, P > 0 → 2 * P = P * (1 + r / 2) ^ 20) ↔ 2 = (1 + r / 2) ^ 20 :=
sorry

end NUMINAMATH_CALUDE_interest_rate_equation_l2098_209837


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l2098_209811

/-- Given a line passing through points (0, -2) and (1, 1), 
    prove that the product of its slope and y-intercept equals -6 -/
theorem line_slope_intercept_product : 
  ∀ (m b : ℝ), 
    (∀ x : ℝ, b = -2 ∧ m * 0 + b = -2) →  -- line passes through (0, -2)
    (∀ x : ℝ, m * 1 + b = 1) →            -- line passes through (1, 1)
    m * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l2098_209811


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2098_209840

theorem inequality_equivalence (x : ℝ) : 
  ‖‖x - 2‖ - 1‖ ≤ 1 ↔ 0 ≤ x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2098_209840


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2098_209841

theorem necessary_not_sufficient_condition (a : ℝ) : 
  (∀ x, ax + 1 = 0 → x^2 + x - 6 = 0) ∧ 
  (∃ x, x^2 + x - 6 = 0 ∧ ax + 1 ≠ 0) →
  a = -1/2 ∨ a = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2098_209841


namespace NUMINAMATH_CALUDE_die_roll_probability_l2098_209815

/-- The probability of rolling a different number on a six-sided die -/
def p_different : ℚ := 5 / 6

/-- The probability of rolling the same number on a six-sided die -/
def p_same : ℚ := 1 / 6

/-- The number of rolls before the final roll -/
def n : ℕ := 9

theorem die_roll_probability :
  p_different ^ n * p_same = (5^8 : ℚ) / (6^9 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_die_roll_probability_l2098_209815


namespace NUMINAMATH_CALUDE_balls_in_boxes_l2098_209842

def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  if k = 2 then
    (n - 1).choose (k - 1) + (n - 2).choose (k - 1)
  else
    0

theorem balls_in_boxes :
  distribute_balls 6 2 = 21 :=
sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l2098_209842


namespace NUMINAMATH_CALUDE_focus_of_parabola_is_correct_l2098_209844

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A parabola in the 2D plane -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ

/-- The given circle from the problem -/
def given_circle : Circle :=
  { center := (3, 0), radius := 4 }

/-- Checks if a point is on the given circle -/
def on_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 16

/-- The parabola from the problem -/
def given_parabola : Parabola :=
  { p := 1, focus := (1, 0) }

/-- Checks if a point is on the given parabola -/
def on_parabola (x y : ℝ) : Prop :=
  y^2 = 2 * given_parabola.p * x

/-- The theorem to be proved -/
theorem focus_of_parabola_is_correct :
  given_parabola.focus = (1, 0) ∧
  given_parabola.p > 0 ∧
  ∃ (x y : ℝ), on_circle x y ∧ on_parabola x y :=
sorry

end NUMINAMATH_CALUDE_focus_of_parabola_is_correct_l2098_209844


namespace NUMINAMATH_CALUDE_fred_earnings_l2098_209814

/-- Represents Fred's chore earnings --/
def chore_earnings (initial_amount final_amount : ℕ) 
  (car_wash_price lawn_mow_price dog_walk_price : ℕ)
  (cars_washed lawns_mowed dogs_walked : ℕ) : Prop :=
  final_amount - initial_amount = 
    car_wash_price * cars_washed + 
    lawn_mow_price * lawns_mowed + 
    dog_walk_price * dogs_walked

/-- Theorem stating that Fred's earnings from chores match the difference in his money --/
theorem fred_earnings :
  chore_earnings 23 86 5 10 3 4 3 7 := by
  sorry

end NUMINAMATH_CALUDE_fred_earnings_l2098_209814


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2098_209885

/-- The area of a square inscribed in an ellipse -/
theorem inscribed_square_area (x y : ℝ) :
  (x^2 / 4 + y^2 / 8 = 1) →  -- Ellipse equation
  (∃ t : ℝ, t > 0 ∧ x = t ∧ y = t) →  -- Square vertex condition
  (4 * t^2 = 32 / 3) :=  -- Area of the square
by
  sorry


end NUMINAMATH_CALUDE_inscribed_square_area_l2098_209885


namespace NUMINAMATH_CALUDE_min_value_w_l2098_209846

/-- The minimum value of w = 2x^2 + 3y^2 + 8x - 5y + 30 is 26.25 -/
theorem min_value_w :
  (∀ x y : ℝ, 2 * x^2 + 3 * y^2 + 8 * x - 5 * y + 30 ≥ 26.25) ∧
  (∃ x y : ℝ, 2 * x^2 + 3 * y^2 + 8 * x - 5 * y + 30 = 26.25) := by
  sorry

end NUMINAMATH_CALUDE_min_value_w_l2098_209846


namespace NUMINAMATH_CALUDE_cubic_extremum_l2098_209829

/-- Given a cubic function f(x) = x³ + 3ax² + bx + a² with an extremum of 0 at x = -1,
    prove that a - b = -7 -/
theorem cubic_extremum (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 + 3*a*x^2 + b*x + a^2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1-ε) (-1+ε), f x ≥ f (-1)) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1-ε) (-1+ε), f x ≤ f (-1)) ∧
  f (-1) = 0 →
  a - b = -7 :=
by sorry


end NUMINAMATH_CALUDE_cubic_extremum_l2098_209829


namespace NUMINAMATH_CALUDE_orchard_apples_count_l2098_209823

theorem orchard_apples_count (total_apples : ℕ) : 
  (40 : ℕ) * total_apples = (100 : ℕ) * (40 : ℕ) * (24 : ℕ) / ((100 : ℕ) - (70 : ℕ)) →
  total_apples = 200 := by
  sorry

end NUMINAMATH_CALUDE_orchard_apples_count_l2098_209823


namespace NUMINAMATH_CALUDE_last_number_is_25_l2098_209879

theorem last_number_is_25 (numbers : Fin 7 → ℝ) :
  (numbers 0 + numbers 1 + numbers 2 + numbers 3) / 4 = 13 →
  (numbers 3 + numbers 4 + numbers 5 + numbers 6) / 4 = 15 →
  numbers 4 + numbers 5 + numbers 6 = 55 →
  (numbers 3) ^ 2 = numbers 6 →
  numbers 6 = 25 := by
sorry

end NUMINAMATH_CALUDE_last_number_is_25_l2098_209879


namespace NUMINAMATH_CALUDE_tan_theta_minus_pi_over_four_l2098_209851

theorem tan_theta_minus_pi_over_four (θ : Real) :
  let z : ℂ := Complex.mk (Real.cos θ - 4/5) (Real.sin θ - 3/5)
  z.re = 0 → Real.tan (θ - Real.pi/4) = -7 :=
by sorry

end NUMINAMATH_CALUDE_tan_theta_minus_pi_over_four_l2098_209851


namespace NUMINAMATH_CALUDE_optimal_scheme_is_best_l2098_209853

/-- Represents a horticultural design scheme -/
structure Scheme where
  a : ℕ  -- number of A type designs
  b : ℕ  -- number of B type designs

/-- Checks if a scheme is feasible given the constraints -/
def is_feasible (s : Scheme) : Prop :=
  s.a + s.b = 50 ∧
  80 * s.a + 50 * s.b ≤ 3490 ∧
  40 * s.a + 90 * s.b ≤ 2950

/-- Calculates the cost of a scheme -/
def cost (s : Scheme) : ℕ :=
  800 * s.a + 960 * s.b

/-- The optimal scheme -/
def optimal_scheme : Scheme :=
  ⟨33, 17⟩

theorem optimal_scheme_is_best :
  is_feasible optimal_scheme ∧
  ∀ s : Scheme, is_feasible s → cost s ≥ cost optimal_scheme :=
sorry

end NUMINAMATH_CALUDE_optimal_scheme_is_best_l2098_209853


namespace NUMINAMATH_CALUDE_power_of_128_l2098_209870

theorem power_of_128 : (128 : ℝ) ^ (4/7 : ℝ) = 16 := by sorry

end NUMINAMATH_CALUDE_power_of_128_l2098_209870


namespace NUMINAMATH_CALUDE_triangle_problem_l2098_209893

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  side_angle_correspondence : True -- This is a placeholder for the correspondence between sides and angles

/-- The theorem statement for the given triangle problem -/
theorem triangle_problem (t : Triangle) :
  (3 * t.a = 2 * t.b) →
  ((t.B = Real.pi / 3 → Real.sin t.C = (Real.sqrt 3 + 3 * Real.sqrt 2) / 6) ∧
   (t.b - t.c = (1 / 3) * t.a → Real.cos t.C = 17 / 27)) := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2098_209893


namespace NUMINAMATH_CALUDE_minor_premise_identification_l2098_209822

-- Define the basic shapes
inductive Shape
| Rectangle
| Parallelogram
| Triangle

-- Define the properties of shapes
def isParallelogram : Shape → Prop
  | Shape.Rectangle => true
  | Shape.Parallelogram => true
  | Shape.Triangle => false

-- Define the syllogism structure
structure Syllogism where
  majorPremise : Prop
  minorPremise : Prop
  conclusion : Prop

-- Define our specific syllogism
def ourSyllogism : Syllogism := {
  majorPremise := isParallelogram Shape.Rectangle
  minorPremise := ¬ isParallelogram Shape.Triangle
  conclusion := Shape.Triangle ≠ Shape.Rectangle
}

-- Theorem to prove
theorem minor_premise_identification :
  ourSyllogism.minorPremise = ¬ isParallelogram Shape.Triangle :=
by sorry

end NUMINAMATH_CALUDE_minor_premise_identification_l2098_209822


namespace NUMINAMATH_CALUDE_no_real_solution_for_sqrt_equation_l2098_209884

theorem no_real_solution_for_sqrt_equation :
  ¬∃ (x : ℝ), Real.sqrt (3 - Real.sqrt x) = 2 := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_for_sqrt_equation_l2098_209884


namespace NUMINAMATH_CALUDE_total_unique_plants_l2098_209849

-- Define the flower beds as finite sets
variable (A B C : Finset ℕ)

-- Define the cardinalities of the sets
variable (card_A : Finset.card A = 600)
variable (card_B : Finset.card B = 500)
variable (card_C : Finset.card C = 400)

-- Define the intersections
variable (card_AB : Finset.card (A ∩ B) = 60)
variable (card_AC : Finset.card (A ∩ C) = 80)
variable (card_BC : Finset.card (B ∩ C) = 40)
variable (card_ABC : Finset.card (A ∩ B ∩ C) = 20)

-- Theorem statement
theorem total_unique_plants :
  Finset.card (A ∪ B ∪ C) = 1340 := by
  sorry

end NUMINAMATH_CALUDE_total_unique_plants_l2098_209849


namespace NUMINAMATH_CALUDE_unique_solution_for_F_l2098_209808

/-- The function F as defined in the problem -/
def F (a b c : ℝ) : ℝ := a * b^3 + c

/-- Theorem stating that -5/19 is the unique solution for a in F(a, 2, 3) = F(a, 3, 8) -/
theorem unique_solution_for_F :
  ∃! a : ℝ, F a 2 3 = F a 3 8 ∧ a = -5/19 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_F_l2098_209808


namespace NUMINAMATH_CALUDE_other_donation_is_100_l2098_209812

/-- Represents the fundraiser for basketball equipment -/
structure Fundraiser where
  goal : ℕ
  bronze_donation : ℕ
  silver_donation : ℕ
  bronze_count : ℕ
  silver_count : ℕ
  other_count : ℕ
  final_day_goal : ℕ

/-- Calculates the amount donated by the family with another status -/
def other_donation (f : Fundraiser) : ℕ :=
  f.goal - (f.bronze_donation * f.bronze_count + f.silver_donation * f.silver_count + f.final_day_goal)

/-- Theorem stating that the family with another status donated $100 -/
theorem other_donation_is_100 (f : Fundraiser)
  (h1 : f.goal = 750)
  (h2 : f.bronze_donation = 25)
  (h3 : f.silver_donation = 50)
  (h4 : f.bronze_count = 10)
  (h5 : f.silver_count = 7)
  (h6 : f.other_count = 1)
  (h7 : f.final_day_goal = 50) :
  other_donation f = 100 := by
  sorry

end NUMINAMATH_CALUDE_other_donation_is_100_l2098_209812


namespace NUMINAMATH_CALUDE_unique_intersection_l2098_209873

/-- The value of m for which the vertical line x = m intersects the parabola x = -4y^2 + 2y + 3 at exactly one point -/
def m : ℚ := 13/4

/-- The equation of the parabola -/
def parabola (y : ℝ) : ℝ := -4 * y^2 + 2 * y + 3

/-- Theorem stating that the vertical line x = m intersects the parabola at exactly one point -/
theorem unique_intersection :
  ∃! y : ℝ, parabola y = m :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l2098_209873


namespace NUMINAMATH_CALUDE_candle_height_relation_l2098_209827

/-- Represents the remaining height of a burning candle -/
def remaining_height (initial_height burning_rate t : ℝ) : ℝ :=
  initial_height - burning_rate * t

/-- Theorem stating the relationship between remaining height and burning time for a specific candle -/
theorem candle_height_relation (h t : ℝ) :
  remaining_height 20 4 t = h ↔ h = 20 - 4 * t := by sorry

end NUMINAMATH_CALUDE_candle_height_relation_l2098_209827


namespace NUMINAMATH_CALUDE_no_valid_a_for_quadratic_l2098_209871

theorem no_valid_a_for_quadratic : ¬∃ (a : ℝ), 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + 2*(a+1)*x₁ - (a-1) = 0) ∧
  (x₂^2 + 2*(a+1)*x₂ - (a-1) = 0) ∧
  ((x₁ > 1 ∧ x₂ < 1) ∨ (x₁ < 1 ∧ x₂ > 1)) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_a_for_quadratic_l2098_209871


namespace NUMINAMATH_CALUDE_fraction_value_at_three_l2098_209895

theorem fraction_value_at_three :
  let x : ℝ := 3
  (x^8 + 8*x^4 + 16) / (x^4 - 4) = 93 := by sorry

end NUMINAMATH_CALUDE_fraction_value_at_three_l2098_209895


namespace NUMINAMATH_CALUDE_gcd_of_117_and_182_l2098_209809

theorem gcd_of_117_and_182 :
  Nat.gcd 117 182 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_117_and_182_l2098_209809


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l2098_209810

theorem least_addition_for_divisibility (n : ℕ) : 
  (∃ (k : ℕ), k > 0 ∧ (821562 + k) % 5 = 0) → 
  (∃ (m : ℕ), m ≥ 3 ∧ (821562 + m) % 5 = 0) ∧ 
  (821562 + 3) % 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l2098_209810


namespace NUMINAMATH_CALUDE_five_digit_palindromes_count_l2098_209876

/-- A function that counts the number of 5-digit palindromes -/
def count_5digit_palindromes : ℕ :=
  let A := 9  -- digits 1 to 9
  let B := 10 -- digits 0 to 9
  let C := 10 -- digits 0 to 9
  A * B * C

/-- Theorem stating that the number of 5-digit palindromes is 900 -/
theorem five_digit_palindromes_count : count_5digit_palindromes = 900 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_palindromes_count_l2098_209876


namespace NUMINAMATH_CALUDE_king_high_school_teachers_l2098_209824

/-- The number of students at King High School -/
def num_students : ℕ := 1500

/-- The number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- The number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 3

/-- The number of students in each class -/
def students_per_class : ℕ := 35

/-- The number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- The number of teachers at King High School -/
def num_teachers : ℕ := 86

theorem king_high_school_teachers : 
  (num_students * classes_per_student) / students_per_class / classes_per_teacher = num_teachers := by
  sorry

end NUMINAMATH_CALUDE_king_high_school_teachers_l2098_209824
