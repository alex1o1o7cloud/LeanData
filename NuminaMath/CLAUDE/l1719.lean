import Mathlib

namespace NUMINAMATH_CALUDE_larger_solution_quadratic_l1719_171936

theorem larger_solution_quadratic (x : ℝ) : 
  x^2 - 7*x - 18 = 0 → x ≤ 9 ∧ (∃ y : ℝ, y ≠ x ∧ y^2 - 7*y - 18 = 0) := by
  sorry

end NUMINAMATH_CALUDE_larger_solution_quadratic_l1719_171936


namespace NUMINAMATH_CALUDE_defect_selection_probability_l1719_171929

/-- Given a set of tubes with defects, calculate the probability of selecting specific defect types --/
theorem defect_selection_probability
  (total_tubes : ℕ)
  (type_a_defects : ℕ)
  (type_b_defects : ℕ)
  (h1 : total_tubes = 50)
  (h2 : type_a_defects = 5)
  (h3 : type_b_defects = 3)
  : ℚ :=
  3 / 490

#check defect_selection_probability

end NUMINAMATH_CALUDE_defect_selection_probability_l1719_171929


namespace NUMINAMATH_CALUDE_range_of_a_l1719_171913

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x, x^2 - (a-1)*x + 1 > 0

def q (a : ℝ) : Prop := ∀ x y, x < y → (a+1)^x < (a+1)^y

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) → 
  ((-1 < a ∧ a ≤ 0) ∨ a ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1719_171913


namespace NUMINAMATH_CALUDE_inequality_holds_iff_m_in_interval_l1719_171900

theorem inequality_holds_iff_m_in_interval :
  ∀ m : ℝ, (∀ x : ℝ, -6 < (2 * x^2 + m * x - 4) / (x^2 - x + 1) ∧ 
    (2 * x^2 + m * x - 4) / (x^2 - x + 1) < 4) ↔ 
  -2 < m ∧ m < 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_m_in_interval_l1719_171900


namespace NUMINAMATH_CALUDE_distinctly_marked_fraction_l1719_171925

/-- Proves that the fraction of a 15 by 24 rectangular region that is distinctly marked is 1/6,
    given that one-third of the rectangle is shaded and half of the shaded area is distinctly marked. -/
theorem distinctly_marked_fraction (length width : ℕ) (shaded_fraction marked_fraction : ℚ) :
  length = 15 →
  width = 24 →
  shaded_fraction = 1/3 →
  marked_fraction = 1/2 →
  (shaded_fraction * marked_fraction : ℚ) = 1/6 :=
by sorry

end NUMINAMATH_CALUDE_distinctly_marked_fraction_l1719_171925


namespace NUMINAMATH_CALUDE_order_of_cube_roots_l1719_171988

theorem order_of_cube_roots (a : ℝ) (x y z : ℝ) 
  (hx : x = (1 + 991 * a) ^ (1/3))
  (hy : y = (1 + 992 * a) ^ (1/3))
  (hz : z = (1 + 993 * a) ^ (1/3))
  (ha : a ≤ 0) : 
  z ≤ y ∧ y ≤ x := by
sorry

end NUMINAMATH_CALUDE_order_of_cube_roots_l1719_171988


namespace NUMINAMATH_CALUDE_fraction_value_at_four_l1719_171982

theorem fraction_value_at_four : 
  let x : ℝ := 4
  (x^6 - 64*x^3 + 512) / (x^3 - 16) = 48 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_at_four_l1719_171982


namespace NUMINAMATH_CALUDE_range_of_a_l1719_171944

theorem range_of_a (x a : ℝ) : 
  (∀ x, (1/2 ≤ x ∧ x ≤ 1) → ¬((x-a)*(x-a-1) > 0)) ∧ 
  (∃ x, ¬(1/2 ≤ x ∧ x ≤ 1) ∧ ¬((x-a)*(x-a-1) > 0)) →
  (0 ≤ a ∧ a ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1719_171944


namespace NUMINAMATH_CALUDE_negation_equivalence_l1719_171927

theorem negation_equivalence :
  (¬ ∀ a : ℝ, a ∈ Set.Icc 0 1 → a^4 + a^2 > 1) ↔
  (∃ a : ℝ, a ∈ Set.Icc 0 1 ∧ a^4 + a^2 ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1719_171927


namespace NUMINAMATH_CALUDE_vitamin_a_daily_serving_l1719_171952

/-- The amount of Vitamin A in each pill (in mg) -/
def vitamin_a_per_pill : ℕ := 50

/-- The number of pills needed for the weekly recommended amount -/
def pills_per_week : ℕ := 28

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The recommended daily serving of Vitamin A (in mg) -/
def recommended_daily_serving : ℕ := (vitamin_a_per_pill * pills_per_week) / days_per_week

theorem vitamin_a_daily_serving :
  recommended_daily_serving = 200 := by
  sorry

end NUMINAMATH_CALUDE_vitamin_a_daily_serving_l1719_171952


namespace NUMINAMATH_CALUDE_b₁_value_l1719_171923

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 8 + 32*x - 12*x^2 - 4*x^3 + x^4

-- Define the set of roots of f(x)
def roots_f : Set ℝ := {x | f x = 0}

-- Define the polynomial g(x)
def g (b₀ b₁ b₂ b₃ : ℝ) (x : ℝ) : ℝ := b₀ + b₁*x + b₂*x^2 + b₃*x^3 + x^4

-- Define the set of roots of g(x)
def roots_g (b₀ b₁ b₂ b₃ : ℝ) : Set ℝ := {x | g b₀ b₁ b₂ b₃ x = 0}

theorem b₁_value (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : roots_f = {x₁, x₂, x₃, x₄})
  (h₂ : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄)
  (h₃ : ∃ b₀ b₁ b₂ b₃, roots_g b₀ b₁ b₂ b₃ = {x₁^2, x₂^2, x₃^2, x₄^2}) :
  ∃ b₀ b₂ b₃, g b₀ (-1024) b₂ b₃ = g b₀ b₁ b₂ b₃ := by sorry

end NUMINAMATH_CALUDE_b₁_value_l1719_171923


namespace NUMINAMATH_CALUDE_equal_marked_cells_exist_l1719_171970

/-- Represents an L-shaped triomino -/
structure Triomino where
  cells : Fin 3 → (Fin 2010 × Fin 2010)

/-- Represents a marking of cells in the grid -/
def Marking := Fin 2010 → Fin 2010 → Bool

/-- Checks if a marking is valid (one cell per triomino) -/
def isValidMarking (grid : List Triomino) (m : Marking) : Prop := sorry

/-- Counts marked cells in a given row -/
def countMarkedInRow (m : Marking) (row : Fin 2010) : Nat := sorry

/-- Counts marked cells in a given column -/
def countMarkedInColumn (m : Marking) (col : Fin 2010) : Nat := sorry

/-- Main theorem statement -/
theorem equal_marked_cells_exist (grid : List Triomino) 
  (h : grid.length = 2010 * 2010 / 3) : 
  ∃ m : Marking, 
    isValidMarking grid m ∧ 
    (∀ r₁ r₂ : Fin 2010, countMarkedInRow m r₁ = countMarkedInRow m r₂) ∧
    (∀ c₁ c₂ : Fin 2010, countMarkedInColumn m c₁ = countMarkedInColumn m c₂) := by
  sorry

end NUMINAMATH_CALUDE_equal_marked_cells_exist_l1719_171970


namespace NUMINAMATH_CALUDE_stars_per_bottle_l1719_171922

/-- Given that Shiela prepared 45 paper stars and has 9 classmates,
    prove that the number of stars per bottle is 5. -/
theorem stars_per_bottle (total_stars : ℕ) (num_classmates : ℕ) 
  (h1 : total_stars = 45) (h2 : num_classmates = 9) :
  total_stars / num_classmates = 5 := by
  sorry

end NUMINAMATH_CALUDE_stars_per_bottle_l1719_171922


namespace NUMINAMATH_CALUDE_family_pizza_order_l1719_171968

/-- Calculates the number of pizzas needed for a family -/
def pizzas_needed (adults : ℕ) (children : ℕ) (adult_slices : ℕ) (child_slices : ℕ) (slices_per_pizza : ℕ) : ℕ :=
  ((adults * adult_slices + children * child_slices) + slices_per_pizza - 1) / slices_per_pizza

/-- Proves that a family of 2 adults and 6 children needs 3 pizzas -/
theorem family_pizza_order : pizzas_needed 2 6 3 1 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_family_pizza_order_l1719_171968


namespace NUMINAMATH_CALUDE_sum_f_at_one_equals_exp_e_l1719_171928

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => fun x => Real.exp x
| (n + 1) => fun x => x * (deriv (f n)) x

theorem sum_f_at_one_equals_exp_e :
  (∑' n, (f n 1) / n.factorial) = Real.exp (Real.exp 1) := by sorry

end NUMINAMATH_CALUDE_sum_f_at_one_equals_exp_e_l1719_171928


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1719_171963

theorem arithmetic_computation : -9 * 5 - (-7 * -2) + (-11 * -4) = -15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1719_171963


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1719_171975

theorem polynomial_factorization (x : ℤ) :
  4 * (x + 4) * (x + 7) * (x + 9) * (x + 11) - 5 * x^2 = (2 * x^2 + 72 * x + 126)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1719_171975


namespace NUMINAMATH_CALUDE_mama_bird_stolen_worms_l1719_171961

/-- The number of worms stolen from Mama bird -/
def stolen_worms : ℕ := by sorry

theorem mama_bird_stolen_worms :
  let babies : ℕ := 6
  let worms_per_baby_per_day : ℕ := 3
  let days : ℕ := 3
  let papa_worms : ℕ := 9
  let mama_worms : ℕ := 13
  let additional_worms_needed : ℕ := 34
  
  stolen_worms = 2 := by sorry

end NUMINAMATH_CALUDE_mama_bird_stolen_worms_l1719_171961


namespace NUMINAMATH_CALUDE_G_properties_l1719_171938

/-- The curve G defined by x³ + y³ - 6xy = 0 for x > 0 and y > 0 -/
def G : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ p.1^3 + p.2^3 - 6*p.1*p.2 = 0}

/-- The line y = x -/
def line_y_eq_x : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = p.2}

/-- The line x + y - 6 = 0 -/
def line_x_plus_y_eq_6 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 6}

theorem G_properties :
  (∀ p : ℝ × ℝ, p ∈ G → (p.2, p.1) ∈ G) ∧ 
  (∃! p : ℝ × ℝ, p ∈ G ∩ line_x_plus_y_eq_6) ∧
  (∀ p : ℝ × ℝ, p ∈ G → Real.sqrt (p.1^2 + p.2^2) ≤ 3 * Real.sqrt 2) ∧
  (∃ p : ℝ × ℝ, p ∈ G ∧ Real.sqrt (p.1^2 + p.2^2) = 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_G_properties_l1719_171938


namespace NUMINAMATH_CALUDE_ali_seashells_l1719_171951

/-- Proves that Ali started with 180 seashells given the conditions of the problem -/
theorem ali_seashells : 
  ∀ S : ℕ, 
  (S - 40 - 30) / 2 = 55 → 
  S = 180 := by
sorry

end NUMINAMATH_CALUDE_ali_seashells_l1719_171951


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l1719_171974

theorem discount_percentage_proof (jacket_price shirt_price : ℝ) 
  (jacket_discount shirt_discount : ℝ) : 
  jacket_price = 100 →
  shirt_price = 50 →
  jacket_discount = 0.3 →
  shirt_discount = 0.6 →
  (jacket_price * jacket_discount + shirt_price * shirt_discount) / (jacket_price + shirt_price) = 0.4 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l1719_171974


namespace NUMINAMATH_CALUDE_hyeyoung_walk_distance_l1719_171965

/-- Given a promenade of length 6 km, prove that walking to its halfway point is 3 km. -/
theorem hyeyoung_walk_distance (promenade_length : ℝ) (hyeyoung_distance : ℝ) 
  (h1 : promenade_length = 6)
  (h2 : hyeyoung_distance = promenade_length / 2) :
  hyeyoung_distance = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyeyoung_walk_distance_l1719_171965


namespace NUMINAMATH_CALUDE_smallest_n_with_four_pairs_l1719_171957

/-- g(n) returns the number of distinct ordered pairs of positive integers (a, b) such that a^2 + b^2 = n -/
def g (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 65 is the smallest positive integer n for which g(n) = 4 -/
theorem smallest_n_with_four_pairs : (∀ m : ℕ, 0 < m → m < 65 → g m ≠ 4) ∧ g 65 = 4 := by sorry

end NUMINAMATH_CALUDE_smallest_n_with_four_pairs_l1719_171957


namespace NUMINAMATH_CALUDE_sin_sum_of_roots_l1719_171914

theorem sin_sum_of_roots (a b c : ℝ) (α β : ℝ) : 
  (∀ x, a * Real.cos x + b * Real.sin x + c = 0 ↔ x = α ∨ x = β) →
  0 < α → α < π →
  0 < β → β < π →
  α ≠ β →
  Real.sin (α + β) = (2 * a * b) / (a^2 + b^2) := by
sorry

end NUMINAMATH_CALUDE_sin_sum_of_roots_l1719_171914


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l1719_171906

/-- Given a cement mixture composed of sand, water, and gravel, where:
    - 1/4 of the mixture is sand (by weight)
    - 2/5 of the mixture is water (by weight)
    - 14 pounds of the mixture is gravel
    Prove that the total weight of the mixture is 40 pounds. -/
theorem cement_mixture_weight :
  ∀ (total_weight : ℝ),
  (1/4 : ℝ) * total_weight +     -- Weight of sand
  (2/5 : ℝ) * total_weight +     -- Weight of water
  14 = total_weight →            -- Weight of gravel
  total_weight = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_cement_mixture_weight_l1719_171906


namespace NUMINAMATH_CALUDE_equation_solution_l1719_171999

theorem equation_solution : ∃! x : ℝ, 4 * x + 9 * x = 430 - 10 * (x + 4) :=
  by
    use 17
    constructor
    · -- Prove that 17 satisfies the equation
      sorry
    · -- Prove that 17 is the unique solution
      sorry

end NUMINAMATH_CALUDE_equation_solution_l1719_171999


namespace NUMINAMATH_CALUDE_triple_hash_45_l1719_171953

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.4 * N + 3

-- State the theorem
theorem triple_hash_45 : hash (hash (hash 45)) = 7.56 := by
  sorry

end NUMINAMATH_CALUDE_triple_hash_45_l1719_171953


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l1719_171948

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 1080 ∣ n^2) : ∃ q : ℕ, q > 0 ∧ q ∣ n ∧ ∀ m : ℕ, m > 0 ∧ m ∣ n → m ≤ q ∧ q = 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l1719_171948


namespace NUMINAMATH_CALUDE_calculate_expression_l1719_171997

def A (n k : ℕ) : ℕ := n * (n - 1) * (n - 2)

def C (n k : ℕ) : ℕ := n * (n - 1) * (n - 2) / (3 * 2 * 1)

theorem calculate_expression : (3 * A 5 3 + 4 * C 6 3) / (3 * 2 * 1) = 130 / 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1719_171997


namespace NUMINAMATH_CALUDE_watch_cost_price_l1719_171958

theorem watch_cost_price (CP : ℝ) : 
  (0.90 * CP = CP - 0.10 * CP) →
  (1.05 * CP = CP + 0.05 * CP) →
  (1.05 * CP - 0.90 * CP = 180) →
  CP = 1200 :=
by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1719_171958


namespace NUMINAMATH_CALUDE_acute_angle_m_range_l1719_171976

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (4, m)

def angle_is_acute (v w : ℝ × ℝ) : Prop :=
  0 < v.1 * w.1 + v.2 * w.2 ∧ 
  (v.1 * w.1 + v.2 * w.2)^2 < (v.1^2 + v.2^2) * (w.1^2 + w.2^2)

theorem acute_angle_m_range :
  ∀ m : ℝ, angle_is_acute a (b m) → m ∈ Set.Ioo (-2) 8 ∪ Set.Ioi 8 :=
by sorry

end NUMINAMATH_CALUDE_acute_angle_m_range_l1719_171976


namespace NUMINAMATH_CALUDE_linear_function_properties_l1719_171985

def f (x : ℝ) := -2 * x + 2

theorem linear_function_properties :
  (∃ (x y : ℝ), f x = y ∧ x > 0 ∧ y > 0) ∧  -- First quadrant
  (∃ (x y : ℝ), f x = y ∧ x < 0 ∧ y > 0) ∧  -- Second quadrant
  (∃ (x y : ℝ), f x = y ∧ x > 0 ∧ y < 0) ∧  -- Fourth quadrant
  (f 2 ≠ 0) ∧                               -- x-intercept is not at (2, 0)
  (∀ x > 0, f x < 2) ∧                      -- When x > 0, y < 2
  (∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂)          -- Function is decreasing
  := by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l1719_171985


namespace NUMINAMATH_CALUDE_tangent_point_and_perpendicular_line_l1719_171943

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the condition for the tangent line being parallel to 4x - y - 1 = 0
def tangent_parallel (x : ℝ) : Prop := f' x = 4

-- Define the third quadrant condition
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Main theorem
theorem tangent_point_and_perpendicular_line :
  ∃ (x₀ y₀ : ℝ), 
    y₀ = f x₀ ∧ 
    tangent_parallel x₀ ∧ 
    third_quadrant x₀ y₀ ∧ 
    x₀ = -1 ∧ 
    y₀ = -4 ∧ 
    ∀ (x y : ℝ), x + 4*y + 17 = 0 ↔ y - y₀ = -(1/4) * (x - x₀) :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_and_perpendicular_line_l1719_171943


namespace NUMINAMATH_CALUDE_flippers_win_probability_l1719_171932

theorem flippers_win_probability :
  let n : ℕ := 6  -- Total number of games
  let k : ℕ := 4  -- Number of games to win
  let p : ℚ := 3/5  -- Probability of winning a single game
  Nat.choose n k * p^k * (1-p)^(n-k) = 4860/15625 := by
sorry

end NUMINAMATH_CALUDE_flippers_win_probability_l1719_171932


namespace NUMINAMATH_CALUDE_victoria_wins_l1719_171902

/-- Represents a player in the game -/
inductive Player : Type
| Harry : Player
| Victoria : Player

/-- Represents a line segment on the grid -/
inductive Segment : Type
| EastWest : Segment
| NorthSouth : Segment

/-- Represents the state of the game -/
structure GameState :=
(turn : Player)
(harry_score : Nat)
(victoria_score : Nat)
(moves : List Segment)

/-- Represents a strategy for a player -/
def Strategy := GameState → Segment

/-- Determines if a move is valid for a given player -/
def valid_move (player : Player) (segment : Segment) : Bool :=
  match player, segment with
  | Player.Harry, Segment.EastWest => true
  | Player.Victoria, Segment.NorthSouth => true
  | _, _ => false

/-- Determines if a move completes a square -/
def completes_square (state : GameState) (segment : Segment) : Bool :=
  sorry -- Implementation details omitted

/-- Applies a move to the game state -/
def apply_move (state : GameState) (segment : Segment) : GameState :=
  sorry -- Implementation details omitted

/-- Determines if the game is over -/
def game_over (state : GameState) : Bool :=
  sorry -- Implementation details omitted

/-- Determines the winner of the game -/
def winner (state : GameState) : Option Player :=
  sorry -- Implementation details omitted

/-- Victoria's winning strategy -/
def victoria_strategy : Strategy :=
  sorry -- Implementation details omitted

/-- Theorem stating that Victoria has a winning strategy -/
theorem victoria_wins :
  ∀ (harry_strategy : Strategy),
  ∃ (final_state : GameState),
  (game_over final_state = true) ∧
  (winner final_state = some Player.Victoria) :=
sorry

end NUMINAMATH_CALUDE_victoria_wins_l1719_171902


namespace NUMINAMATH_CALUDE_square_area_on_line_and_parabola_l1719_171972

/-- A square with one side on y = x + 4 and two vertices on y² = x has area 18 or 50 -/
theorem square_area_on_line_and_parabola :
  ∀ (A B C D : ℝ × ℝ),
    (∃ (y₁ y₂ : ℝ),
      A.2 = A.1 + 4 ∧
      B.2 = B.1 + 4 ∧
      C = (y₁^2, y₁) ∧
      D = (y₂^2, y₂) ∧
      (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧
      (C.1 - B.1)^2 + (C.2 - B.2)^2 = (D.1 - C.1)^2 + (D.2 - C.2)^2 ∧
      (D.1 - C.1)^2 + (D.2 - C.2)^2 = (A.1 - D.1)^2 + (A.2 - D.2)^2) →
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 18) ∨ ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 50) :=
by sorry


end NUMINAMATH_CALUDE_square_area_on_line_and_parabola_l1719_171972


namespace NUMINAMATH_CALUDE_max_m_value_l1719_171960

theorem max_m_value (m : ℝ) : 
  (¬ ∃ x : ℝ, x ≥ 3 ∧ 2*x - 1 < m) → m ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l1719_171960


namespace NUMINAMATH_CALUDE_santinos_fruits_l1719_171981

/-- The number of papaya trees Santino has -/
def papaya_trees : ℕ := 2

/-- The number of mango trees Santino has -/
def mango_trees : ℕ := 3

/-- The number of papayas produced by each papaya tree -/
def papayas_per_tree : ℕ := 10

/-- The number of mangos produced by each mango tree -/
def mangos_per_tree : ℕ := 20

/-- The total number of fruits Santino has -/
def total_fruits : ℕ := papaya_trees * papayas_per_tree + mango_trees * mangos_per_tree

theorem santinos_fruits : total_fruits = 80 := by
  sorry

end NUMINAMATH_CALUDE_santinos_fruits_l1719_171981


namespace NUMINAMATH_CALUDE_spade_sum_equals_six_l1719_171937

-- Define the ♠ operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_sum_equals_six : 
  (spade 2 3) + (spade 5 10) = 6 := by
  sorry

end NUMINAMATH_CALUDE_spade_sum_equals_six_l1719_171937


namespace NUMINAMATH_CALUDE_arithmetic_geometric_intersection_l1719_171909

/-- An arithmetic sequence of integers -/
def ArithmeticSequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℤ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = r * b n

/-- The theorem statement -/
theorem arithmetic_geometric_intersection (a : ℕ → ℤ) (d : ℤ) (n₁ : ℕ) :
  d ≠ 0 →
  ArithmeticSequence a d →
  a 5 = 6 →
  5 < n₁ →
  GeometricSequence (fun n ↦ if n = 1 then a 3 else if n = 2 then a 5 else a (n₁ + n - 3)) →
  (∃ k : ℕ, k ≤ 7 ∧
    ∀ n : ℕ, n ≤ 2015 →
      (∃ m : ℕ, m ≤ k ∧ a n = if m = 1 then a 3 else if m = 2 then a 5 else a (n₁ + m - 3))) ∧
  (∀ k : ℕ, k > 7 →
    ¬∀ n : ℕ, n ≤ 2015 →
      (∃ m : ℕ, m ≤ k ∧ a n = if m = 1 then a 3 else if m = 2 then a 5 else a (n₁ + m - 3))) :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_geometric_intersection_l1719_171909


namespace NUMINAMATH_CALUDE_johns_total_time_l1719_171947

theorem johns_total_time (exploring_time writing_book_time : ℝ) : 
  exploring_time = 3 →
  writing_book_time = 0.5 →
  exploring_time + (exploring_time / 2) + writing_book_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_johns_total_time_l1719_171947


namespace NUMINAMATH_CALUDE_min_magnitude_a_plus_tb_collinear_a_minus_tb_c_l1719_171977

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (-3, 2)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (3, -1)

/-- The squared magnitude of a vector -/
def magnitude_squared (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

/-- Theorem: Minimum value of |a+tb| and its corresponding t -/
theorem min_magnitude_a_plus_tb :
  (∃ t : ℝ, magnitude_squared (a.1 + t * b.1, a.2 + t * b.2) = (7 * Real.sqrt 5 / 5)^2) ∧
  (∀ t : ℝ, magnitude_squared (a.1 + t * b.1, a.2 + t * b.2) ≥ (7 * Real.sqrt 5 / 5)^2) ∧
  (magnitude_squared (a.1 + 4/5 * b.1, a.2 + 4/5 * b.2) = (7 * Real.sqrt 5 / 5)^2) :=
sorry

/-- Theorem: Value of t when a-tb is collinear with c -/
theorem collinear_a_minus_tb_c :
  ∃ t : ℝ, t = 3/5 ∧ (a.1 - t * b.1) * c.2 = (a.2 - t * b.2) * c.1 :=
sorry

end NUMINAMATH_CALUDE_min_magnitude_a_plus_tb_collinear_a_minus_tb_c_l1719_171977


namespace NUMINAMATH_CALUDE_kiwis_to_add_for_orange_percentage_l1719_171991

/-- Proves that adding 7 kiwis to a box with 24 oranges, 30 kiwis, 15 apples, and 20 bananas
    will make oranges exactly 25% of the total fruits -/
theorem kiwis_to_add_for_orange_percentage (oranges kiwis apples bananas : ℕ) 
    (h1 : oranges = 24) 
    (h2 : kiwis = 30) 
    (h3 : apples = 15) 
    (h4 : bananas = 20) : 
    let total := oranges + kiwis + apples + bananas + 7
    (oranges : ℚ) / total = 1/4 := by sorry

end NUMINAMATH_CALUDE_kiwis_to_add_for_orange_percentage_l1719_171991


namespace NUMINAMATH_CALUDE_frank_candy_purchase_l1719_171930

/-- The number of candies Frank can buy with his arcade tickets -/
def candies_bought (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) : ℕ :=
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost

/-- Theorem: Frank can buy 7 candies with his arcade tickets -/
theorem frank_candy_purchase :
  candies_bought 33 9 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_frank_candy_purchase_l1719_171930


namespace NUMINAMATH_CALUDE_second_number_existence_and_uniqueness_l1719_171993

theorem second_number_existence_and_uniqueness :
  ∃! x : ℕ, x > 0 ∧ 220070 = (555 + x) * (2 * (x - 555)) + 70 :=
by sorry

end NUMINAMATH_CALUDE_second_number_existence_and_uniqueness_l1719_171993


namespace NUMINAMATH_CALUDE_brittany_test_average_l1719_171994

def test_average (score1 : ℚ) (score2 : ℚ) : ℚ :=
  (score1 + score2) / 2

theorem brittany_test_average :
  test_average 78 84 = 81 := by
  sorry

end NUMINAMATH_CALUDE_brittany_test_average_l1719_171994


namespace NUMINAMATH_CALUDE_cube_root_y_fourth_root_y_five_eq_four_l1719_171962

theorem cube_root_y_fourth_root_y_five_eq_four (y : ℝ) :
  (y * (y^5)^(1/4))^(1/3) = 4 → y = 2^(8/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_y_fourth_root_y_five_eq_four_l1719_171962


namespace NUMINAMATH_CALUDE_semicircular_plot_radius_l1719_171964

/-- The radius of a semicircular plot given the total fence length and opening length. -/
theorem semicircular_plot_radius 
  (total_fence_length : ℝ) 
  (opening_length : ℝ) 
  (h1 : total_fence_length = 33) 
  (h2 : opening_length = 3) : 
  ∃ (radius : ℝ), radius = (total_fence_length - opening_length) / (Real.pi + 2) :=
sorry

end NUMINAMATH_CALUDE_semicircular_plot_radius_l1719_171964


namespace NUMINAMATH_CALUDE_sams_initial_dimes_l1719_171939

/-- The problem of determining Sam's initial number of dimes -/
theorem sams_initial_dimes : 
  ∀ (initial final given : ℕ), 
  given = 7 →                 -- Sam's dad gave him 7 dimes
  final = 16 →                -- After receiving the dimes, Sam has 16 dimes
  final = initial + given →   -- The final amount is the sum of initial and given
  initial = 9 :=              -- Prove that the initial amount was 9 dimes
by sorry

end NUMINAMATH_CALUDE_sams_initial_dimes_l1719_171939


namespace NUMINAMATH_CALUDE_max_profit_theorem_l1719_171989

/-- Represents the daily production and profit of an eco-friendly bag factory --/
structure BagFactory where
  totalBags : ℕ
  costA : ℚ
  sellA : ℚ
  costB : ℚ
  sellB : ℚ
  maxInvestment : ℚ

/-- Calculates the profit function for the bag factory --/
def profitFunction (factory : BagFactory) (x : ℚ) : ℚ :=
  (factory.sellA - factory.costA) * x + (factory.sellB - factory.costB) * (factory.totalBags - x)

/-- Theorem stating the maximum profit of the bag factory --/
theorem max_profit_theorem (factory : BagFactory) 
    (h1 : factory.totalBags = 4500)
    (h2 : factory.costA = 2)
    (h3 : factory.sellA = 2.3)
    (h4 : factory.costB = 3)
    (h5 : factory.sellB = 3.5)
    (h6 : factory.maxInvestment = 10000) :
    ∃ x : ℚ, x ≥ 0 ∧ x ≤ factory.totalBags ∧
    factory.costA * x + factory.costB * (factory.totalBags - x) ≤ factory.maxInvestment ∧
    ∀ y : ℚ, y ≥ 0 → y ≤ factory.totalBags →
    factory.costA * y + factory.costB * (factory.totalBags - y) ≤ factory.maxInvestment →
    profitFunction factory x ≥ profitFunction factory y ∧
    profitFunction factory x = 1550 := by
  sorry


end NUMINAMATH_CALUDE_max_profit_theorem_l1719_171989


namespace NUMINAMATH_CALUDE_complex_circle_equation_l1719_171949

open Complex

theorem complex_circle_equation (z : ℂ) (x y : ℝ) :
  z = x + y * I →
  abs (z - 2) = 1 →
  (x - 2)^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_circle_equation_l1719_171949


namespace NUMINAMATH_CALUDE_alternating_color_probability_l1719_171926

/-- The number of white balls in the box -/
def white_balls : ℕ := 5

/-- The number of black balls in the box -/
def black_balls : ℕ := 5

/-- The total number of balls in the box -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of successful alternating sequences -/
def successful_sequences : ℕ := 2

/-- The probability of drawing all balls with alternating colors -/
def alternating_probability : ℚ := successful_sequences / (total_balls.choose white_balls)

theorem alternating_color_probability :
  alternating_probability = 1 / 126 := by
  sorry

end NUMINAMATH_CALUDE_alternating_color_probability_l1719_171926


namespace NUMINAMATH_CALUDE_rectangle_only_convex_four_right_angles_l1719_171940

/-- A polygon is a set of points in the plane -/
def Polygon : Type := Set (ℝ × ℝ)

/-- A polygon is convex if for any two points in the polygon, the line segment between them is entirely contained within the polygon -/
def is_convex (p : Polygon) : Prop := sorry

/-- The number of sides in a polygon -/
def num_sides (p : Polygon) : ℕ := sorry

/-- The number of right angles in a polygon -/
def num_right_angles (p : Polygon) : ℕ := sorry

/-- A rectangle is a polygon with exactly four sides and four right angles -/
def is_rectangle (p : Polygon) : Prop :=
  num_sides p = 4 ∧ num_right_angles p = 4

theorem rectangle_only_convex_four_right_angles (p : Polygon) :
  is_convex p ∧ num_right_angles p = 4 → is_rectangle p :=
sorry

end NUMINAMATH_CALUDE_rectangle_only_convex_four_right_angles_l1719_171940


namespace NUMINAMATH_CALUDE_total_cost_is_2200_l1719_171996

/-- The total cost of buying one smartphone, one personal computer, and one advanced tablet -/
def total_cost (smartphone_price : ℕ) (pc_price_difference : ℕ) : ℕ :=
  let pc_price := smartphone_price + pc_price_difference
  let tablet_price := smartphone_price + pc_price
  smartphone_price + pc_price + tablet_price

/-- Proof that the total cost is $2200 given the specified prices -/
theorem total_cost_is_2200 :
  total_cost 300 500 = 2200 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_2200_l1719_171996


namespace NUMINAMATH_CALUDE_smallest_palindromic_n_string_l1719_171910

/-- An n-string is a string of digits formed by writing the numbers 1, 2, ..., n in some order -/
def nString (n : ℕ) := List ℕ

/-- A palindromic string reads the same forwards and backwards -/
def isPalindromic (s : List ℕ) : Prop :=
  s = s.reverse

/-- The smallest n > 1 such that there exists a palindromic n-string -/
def smallestPalindromicN : ℕ := 19

theorem smallest_palindromic_n_string : 
  (∀ k : ℕ, 1 < k → k < smallestPalindromicN → 
    ¬∃ s : nString k, isPalindromic s) ∧
  (∃ s : nString smallestPalindromicN, isPalindromic s) := by
  sorry

end NUMINAMATH_CALUDE_smallest_palindromic_n_string_l1719_171910


namespace NUMINAMATH_CALUDE_floor_of_negative_three_point_seven_l1719_171941

theorem floor_of_negative_three_point_seven :
  ⌊(-3.7 : ℝ)⌋ = -4 := by sorry

end NUMINAMATH_CALUDE_floor_of_negative_three_point_seven_l1719_171941


namespace NUMINAMATH_CALUDE_remaining_money_l1719_171946

def savings : ℕ := 5555 -- in base 8
def ticket_cost : ℕ := 1200 -- in base 10

def base_8_to_10 (n : ℕ) : ℕ :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem remaining_money :
  base_8_to_10 savings - ticket_cost = 1725 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l1719_171946


namespace NUMINAMATH_CALUDE_interview_probability_implies_total_workers_l1719_171950

/-- The number of workers excluding Jack and Jill -/
def other_workers : ℕ := 6

/-- The probability of selecting both Jack and Jill for the interview -/
def probability : ℚ := 1 / 28

/-- The number of workers to be selected for the interview -/
def selected_workers : ℕ := 2

/-- The total number of workers -/
def total_workers : ℕ := other_workers + 2

theorem interview_probability_implies_total_workers :
  (probability = (1 : ℚ) / (total_workers.choose selected_workers)) →
  total_workers = 8 := by
  sorry

end NUMINAMATH_CALUDE_interview_probability_implies_total_workers_l1719_171950


namespace NUMINAMATH_CALUDE_smallest_common_factor_thirty_three_satisfies_smallest_n_is_33_l1719_171904

theorem smallest_common_factor (n : ℕ) : n > 0 ∧ ∃ (k : ℕ), k > 1 ∧ k ∣ (8*n - 3) ∧ k ∣ (6*n + 5) → n ≥ 33 :=
sorry

theorem thirty_three_satisfies : ∃ (k : ℕ), k > 1 ∧ k ∣ (8*33 - 3) ∧ k ∣ (6*33 + 5) :=
sorry

theorem smallest_n_is_33 : (∃ (n : ℕ), n > 0 ∧ ∃ (k : ℕ), k > 1 ∧ k ∣ (8*n - 3) ∧ k ∣ (6*n + 5)) ∧
  (∀ (m : ℕ), m > 0 ∧ ∃ (k : ℕ), k > 1 ∧ k ∣ (8*m - 3) ∧ k ∣ (6*m + 5) → m ≥ 33) :=
sorry

end NUMINAMATH_CALUDE_smallest_common_factor_thirty_three_satisfies_smallest_n_is_33_l1719_171904


namespace NUMINAMATH_CALUDE_no_solution_exists_l1719_171911

theorem no_solution_exists : 
  ¬∃ (a b : ℕ+), a * b + 82 = 25 * Nat.lcm a b + 15 * Nat.gcd a b :=
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1719_171911


namespace NUMINAMATH_CALUDE_empty_solution_set_iff_a_in_range_l1719_171924

-- Define the quadratic function
def f (a x : ℝ) : ℝ := (a - 1) * x^2 + 2 * (a - 1) * x - 4

-- Define the property of empty solution set
def has_empty_solution_set (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x < 0

-- State the theorem
theorem empty_solution_set_iff_a_in_range :
  ∀ a : ℝ, has_empty_solution_set a ↔ -3 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_empty_solution_set_iff_a_in_range_l1719_171924


namespace NUMINAMATH_CALUDE_no_four_distinct_squares_sum_to_100_l1719_171916

theorem no_four_distinct_squares_sum_to_100 : 
  ¬ ∃ (a b c d : ℕ), 
    (0 < a) ∧ (a < b) ∧ (b < c) ∧ (c < d) ∧ 
    (a^2 + b^2 + c^2 + d^2 = 100) :=
sorry

end NUMINAMATH_CALUDE_no_four_distinct_squares_sum_to_100_l1719_171916


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1719_171933

theorem max_value_quadratic (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x + 1) :
  ∃ y ∈ Set.Icc (-2 : ℝ) 2, ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≤ f y ∧ f y = 9 :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1719_171933


namespace NUMINAMATH_CALUDE_work_ratio_l1719_171983

/-- Given that A can finish a work in 12 days and A and B together can finish 0.25 part of the work in a day,
    prove that the ratio of time taken by B to finish the work alone to the time taken by A is 1:2 -/
theorem work_ratio (time_A : ℝ) (combined_rate : ℝ) :
  time_A = 12 →
  combined_rate = 0.25 →
  combined_rate = 1 / time_A + 1 / (time_A / 2) :=
by sorry

end NUMINAMATH_CALUDE_work_ratio_l1719_171983


namespace NUMINAMATH_CALUDE_dividend_calculation_l1719_171978

theorem dividend_calculation (divisor quotient remainder dividend : ℕ) : 
  divisor = 13 → quotient = 17 → remainder = 1 → 
  dividend = divisor * quotient + remainder →
  dividend = 222 := by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1719_171978


namespace NUMINAMATH_CALUDE_vacation_cost_per_person_l1719_171905

theorem vacation_cost_per_person (num_people : ℕ) (airbnb_cost car_cost : ℚ) :
  num_people = 8 ∧ airbnb_cost = 3200 ∧ car_cost = 800 →
  (airbnb_cost + car_cost) / num_people = 500 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_per_person_l1719_171905


namespace NUMINAMATH_CALUDE_reflection_result_l1719_171934

/-- Reflects a point (x, y) across the line x = k -/
def reflect_point (x y k : ℝ) : ℝ × ℝ := (2 * k - x, y)

/-- Reflects a line y = mx + c across x = k -/
def reflect_line (m c k : ℝ) : ℝ × ℝ := 
  let point := reflect_point k (m * k + c) k
  (-m, 2 * m * k + c - m * point.1)

theorem reflection_result : 
  let original_slope : ℝ := -2
  let original_intercept : ℝ := 7
  let reflection_line : ℝ := 3
  let (a, b) := reflect_line original_slope original_intercept reflection_line
  2 * a + b = -1 := by sorry

end NUMINAMATH_CALUDE_reflection_result_l1719_171934


namespace NUMINAMATH_CALUDE_fifth_term_of_special_arithmetic_sequence_l1719_171908

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem fifth_term_of_special_arithmetic_sequence (seq : ArithmeticSequence) 
    (h1 : seq.a 1 = 2)
    (h2 : 3 * seq.S 3 = seq.S 2 + seq.S 4) : 
  seq.a 5 = -10 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_special_arithmetic_sequence_l1719_171908


namespace NUMINAMATH_CALUDE_survey_students_l1719_171995

theorem survey_students (total_allowance : ℚ) 
  (h1 : total_allowance = 320)
  (h2 : (2 : ℚ) / 3 * 6 + (1 : ℚ) / 3 * 4 = 16 / 3) : 
  ∃ (num_students : ℕ), num_students * (16 : ℚ) / 3 = total_allowance ∧ num_students = 60 := by
  sorry

end NUMINAMATH_CALUDE_survey_students_l1719_171995


namespace NUMINAMATH_CALUDE_percentage_difference_l1719_171959

theorem percentage_difference : 
  (80 / 100 * 40) - (4 / 5 * 15) = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1719_171959


namespace NUMINAMATH_CALUDE_tangent_circle_equations_l1719_171920

def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 4 = 0

def tangent_line (y : ℝ) : Prop :=
  y = 0

def is_tangent_circles (x1 y1 r1 x2 y2 r2 : ℝ) : Prop :=
  (x1 - x2)^2 + (y1 - y2)^2 = (r1 + r2)^2 ∨ (x1 - x2)^2 + (y1 - y2)^2 = (r1 - r2)^2

def is_tangent_circle_line (x y r : ℝ) : Prop :=
  y = r ∨ y = -r

theorem tangent_circle_equations :
  ∃ (a b c d : ℝ),
    (∀ x y : ℝ, ((x - (2 + 2 * Real.sqrt 10))^2 + (y - 4)^2 = 16 ↔ (x - a)^2 + (y - 4)^2 = 16) ∧
                ((x - (2 - 2 * Real.sqrt 10))^2 + (y - 4)^2 = 16 ↔ (x - b)^2 + (y - 4)^2 = 16) ∧
                ((x - (2 + 2 * Real.sqrt 6))^2 + (y + 4)^2 = 16 ↔ (x - c)^2 + (y + 4)^2 = 16) ∧
                ((x - (2 - 2 * Real.sqrt 6))^2 + (y + 4)^2 = 16 ↔ (x - d)^2 + (y + 4)^2 = 16)) ∧
    (∀ x y : ℝ, given_circle x y →
      (is_tangent_circles x y 3 a 4 4 ∧ is_tangent_circle_line a 4 4) ∨
      (is_tangent_circles x y 3 b 4 4 ∧ is_tangent_circle_line b 4 4) ∨
      (is_tangent_circles x y 3 c (-4) 4 ∧ is_tangent_circle_line c (-4) 4) ∨
      (is_tangent_circles x y 3 d (-4) 4 ∧ is_tangent_circle_line d (-4) 4)) ∧
    (∀ x y : ℝ, tangent_line y →
      ((x - a)^2 + (y - 4)^2 = 16 ∨
       (x - b)^2 + (y - 4)^2 = 16 ∨
       (x - c)^2 + (y + 4)^2 = 16 ∨
       (x - d)^2 + (y + 4)^2 = 16)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_equations_l1719_171920


namespace NUMINAMATH_CALUDE_discounted_price_approx_l1719_171987

/-- The original price of the shirt in rupees -/
def original_price : ℝ := 746.67

/-- The discount percentage as a decimal -/
def discount_rate : ℝ := 0.25

/-- The discounted price of the shirt -/
def discounted_price : ℝ := original_price * (1 - discount_rate)

/-- Theorem stating that the discounted price is approximately 560 rupees -/
theorem discounted_price_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |discounted_price - 560| < ε :=
sorry

end NUMINAMATH_CALUDE_discounted_price_approx_l1719_171987


namespace NUMINAMATH_CALUDE_power_division_l1719_171969

theorem power_division (m : ℝ) : m^10 / m^5 = m^5 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l1719_171969


namespace NUMINAMATH_CALUDE_jenny_games_against_mark_l1719_171912

theorem jenny_games_against_mark (mark_wins : ℕ) (jenny_wins : ℕ) 
  (h1 : mark_wins = 1)
  (h2 : jenny_wins = 14) :
  ∃ m : ℕ,
    (m - mark_wins) + (2 * m - (3/4 * 2 * m)) = jenny_wins ∧ 
    m = 30 := by
  sorry

end NUMINAMATH_CALUDE_jenny_games_against_mark_l1719_171912


namespace NUMINAMATH_CALUDE_boots_discounted_price_l1719_171956

/-- Calculates the discounted price of an item given its original price and discount percentage. -/
def discountedPrice (originalPrice : ℚ) (discountPercentage : ℚ) : ℚ :=
  originalPrice * (1 - discountPercentage / 100)

/-- Proves that the discounted price of boots with an original price of $90 and a 20% discount is $72. -/
theorem boots_discounted_price :
  discountedPrice 90 20 = 72 := by
  sorry

end NUMINAMATH_CALUDE_boots_discounted_price_l1719_171956


namespace NUMINAMATH_CALUDE_smallest_special_number_l1719_171980

/-- A number is composite if it's not prime -/
def IsComposite (n : ℕ) : Prop := ¬ Nat.Prime n

/-- A number has no prime factor less than m if all its prime factors are greater than or equal to m -/
def NoPrimeFactorLessThan (n m : ℕ) : Prop :=
  ∀ p, p < m → Nat.Prime p → ¬(p ∣ n)

theorem smallest_special_number : ∃ n : ℕ,
  n > 3000 ∧
  IsComposite n ∧
  ¬(∃ k : ℕ, n = k^2) ∧
  NoPrimeFactorLessThan n 60 ∧
  (∀ m : ℕ, m > 3000 → IsComposite m → ¬(∃ k : ℕ, m = k^2) → NoPrimeFactorLessThan m 60 → m ≥ n) ∧
  n = 4087 := by
sorry

end NUMINAMATH_CALUDE_smallest_special_number_l1719_171980


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1719_171931

/-- Represents a hyperbola with focus on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_relation : a^2 + b^2 = c^2

/-- The standard equation of a hyperbola -/
def standardEquation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The equation of an asymptote of a hyperbola -/
def asymptoteEquation (h : Hyperbola) (x y : ℝ) : Prop :=
  y = (h.b / h.a) * x

theorem hyperbola_equation (h : Hyperbola) 
  (h_asymptote : asymptoteEquation h x y ↔ y = 2 * x)
  (h_focus : h.c = Real.sqrt 5) :
  standardEquation h x y ↔ x^2 - y^2 / 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1719_171931


namespace NUMINAMATH_CALUDE_complement_P_equals_two_l1719_171966

def U : Set Int := {-1, 0, 1, 2}

def P : Set Int := {x : Int | x^2 < 2}

theorem complement_P_equals_two : 
  {x ∈ U | x ∉ P} = {2} := by sorry

end NUMINAMATH_CALUDE_complement_P_equals_two_l1719_171966


namespace NUMINAMATH_CALUDE_initial_alcohol_content_75_percent_l1719_171954

/-- Represents the alcohol content of a solution as a real number between 0 and 1 -/
def AlcoholContent := { x : ℝ // 0 ≤ x ∧ x ≤ 1 }

/-- Proves that the initial alcohol content was 75% given the problem conditions -/
theorem initial_alcohol_content_75_percent 
  (initial_volume : ℝ) 
  (drained_volume : ℝ) 
  (added_content : AlcoholContent) 
  (final_content : AlcoholContent) 
  (h1 : initial_volume = 1)
  (h2 : drained_volume = 0.4)
  (h3 : added_content.val = 0.5)
  (h4 : final_content.val = 0.65) :
  ∃ (initial_content : AlcoholContent), 
    initial_content.val = 0.75 ∧
    (initial_volume - drained_volume) * initial_content.val + 
    drained_volume * added_content.val = 
    initial_volume * final_content.val :=
by sorry


end NUMINAMATH_CALUDE_initial_alcohol_content_75_percent_l1719_171954


namespace NUMINAMATH_CALUDE_goldbach_2024_l1719_171973

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem goldbach_2024 :
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 2024 :=
sorry

end NUMINAMATH_CALUDE_goldbach_2024_l1719_171973


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l1719_171935

/-- Given a college with 416 total students and 160 girls, the ratio of boys to girls is 8:5 -/
theorem boys_to_girls_ratio (total_students : ℕ) (girls : ℕ) : 
  total_students = 416 → girls = 160 → 
  (total_students - girls) / girls = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l1719_171935


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_a_plus_2b_exact_min_value_a_plus_2b_equality_l1719_171907

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 2) + 1 / (b + 2) = 1 / 3) : 
  ∀ x y, x > 0 → y > 0 → 1 / (x + 2) + 1 / (y + 2) = 1 / 3 → a + 2 * b ≤ x + 2 * y :=
by sorry

theorem min_value_a_plus_2b_exact (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 2) + 1 / (b + 2) = 1 / 3) : 
  a + 2 * b ≥ 3 + 6 * Real.sqrt 2 :=
by sorry

theorem min_value_a_plus_2b_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 2) + 1 / (b + 2) = 1 / 3) : 
  (a + 2 * b = 3 + 6 * Real.sqrt 2) ↔ 
  (a = 1 + 3 * Real.sqrt 2 ∧ b = 1 + 3 * Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_a_plus_2b_exact_min_value_a_plus_2b_equality_l1719_171907


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l1719_171903

theorem circle_diameter_from_area :
  ∀ (r : ℝ), r > 0 → π * r^2 = 150 * π → 2 * r = 10 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l1719_171903


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l1719_171945

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) := a * x^2 + 5 * x - 2

-- Define the solution set condition
def solution_set_condition (a : ℝ) : Prop :=
  ∀ x, f a x > 0 ↔ (1/2 < x ∧ x < 2)

-- Theorem statement
theorem quadratic_inequality_problem (a : ℝ) (h : solution_set_condition a) :
  a = -2 ∧ 
  (∀ x, a * x^2 + 5 * x + a^2 - 1 > 0 ↔ (-1/2 < x ∧ x < 3)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l1719_171945


namespace NUMINAMATH_CALUDE_max_sections_five_lines_l1719_171919

/-- The maximum number of sections a rectangle can be divided into by n line segments --/
def max_sections (n : ℕ) : ℕ :=
  if n = 0 then 1 else (n * (n + 1)) / 2 + 1

/-- Theorem: The maximum number of sections a rectangle can be divided into by 5 line segments is 16 --/
theorem max_sections_five_lines :
  max_sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sections_five_lines_l1719_171919


namespace NUMINAMATH_CALUDE_prove_last_score_l1719_171992

def scores : List ℤ := [50, 55, 60, 85, 90, 100]

def is_integer_average (sublist : List ℤ) : Prop :=
  ∃ n : ℤ, (sublist.sum : ℚ) / sublist.length = n

def last_score_is_60 : Prop :=
  ∀ perm : List ℤ, perm.length = 6 →
    perm.toFinset = scores.toFinset →
    (∀ k : ℕ, k ≤ 5 → is_integer_average (perm.take k)) →
    perm.reverse.head? = some 60

theorem prove_last_score : last_score_is_60 := by
  sorry

end NUMINAMATH_CALUDE_prove_last_score_l1719_171992


namespace NUMINAMATH_CALUDE_product_is_even_l1719_171990

def pi_digits : Finset ℕ := {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4}

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem product_is_even (a : Fin 24 → ℕ) (h : ∀ i, a i ∈ pi_digits) :
  is_even ((a 0 - a 1) * (a 2 - a 3) * (a 4 - a 5) * (a 6 - a 7) * (a 8 - a 9) * (a 10 - a 11) *
           (a 12 - a 13) * (a 14 - a 15) * (a 16 - a 17) * (a 18 - a 19) * (a 20 - a 21) * (a 22 - a 23)) :=
by sorry

end NUMINAMATH_CALUDE_product_is_even_l1719_171990


namespace NUMINAMATH_CALUDE_sqrt_400_divided_by_2_l1719_171942

theorem sqrt_400_divided_by_2 : Real.sqrt 400 / 2 = 10 := by sorry

end NUMINAMATH_CALUDE_sqrt_400_divided_by_2_l1719_171942


namespace NUMINAMATH_CALUDE_root_equation_problem_l1719_171955

/-- Given two equations with constants p and q, prove that p = 5, q = -10, and 50p + q = 240 -/
theorem root_equation_problem (p q : ℝ) : 
  (∃! x y : ℝ, x ≠ y ∧ ((x + p) * (x + q) * (x - 8) = 0 ∨ x = 5)) →
  (∃! x y : ℝ, x ≠ y ∧ ((x + 2*p) * (x - 5) * (x - 10) = 0 ∨ x = -q ∨ x = 8)) →
  p = 5 ∧ q = -10 ∧ 50*p + q = 240 := by
sorry


end NUMINAMATH_CALUDE_root_equation_problem_l1719_171955


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1719_171971

theorem quadratic_inequality (x : ℝ) : x^2 - 6*x - 16 > 0 ↔ x < -2 ∨ x > 8 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1719_171971


namespace NUMINAMATH_CALUDE_only_cone_cannot_have_quadrilateral_cross_section_l1719_171918

-- Define the types of solids
inductive Solid
  | Cylinder
  | Cone
  | FrustumOfCone
  | Prism

-- Define a function that checks if a solid can have a quadrilateral cross-section
def canHaveQuadrilateralCrossSection (s : Solid) : Prop :=
  match s with
  | Solid.Cylinder => true
  | Solid.Cone => false
  | Solid.FrustumOfCone => true
  | Solid.Prism => true

-- Theorem stating that only a Cone cannot have a quadrilateral cross-section
theorem only_cone_cannot_have_quadrilateral_cross_section :
  ∀ s : Solid, ¬(canHaveQuadrilateralCrossSection s) ↔ s = Solid.Cone :=
by
  sorry


end NUMINAMATH_CALUDE_only_cone_cannot_have_quadrilateral_cross_section_l1719_171918


namespace NUMINAMATH_CALUDE_cost_of_paints_l1719_171901

def cost_paintbrush : ℚ := 2.40
def cost_easel : ℚ := 6.50
def rose_has : ℚ := 7.10
def rose_needs : ℚ := 11.00

theorem cost_of_paints :
  let total_cost := rose_has + rose_needs
  let cost_paints := total_cost - (cost_paintbrush + cost_easel)
  cost_paints = 9.20 := by sorry

end NUMINAMATH_CALUDE_cost_of_paints_l1719_171901


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l1719_171986

theorem polynomial_identity_sum (b₁ b₂ b₃ b₄ c₁ c₂ c₃ c₄ : ℝ) :
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃) * (x^2 + b₄*x + c₄)) →
  b₁*c₁ + b₂*c₂ + b₃*c₃ + b₄*c₄ = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l1719_171986


namespace NUMINAMATH_CALUDE_students_with_d_grade_l1719_171984

/-- Proves that in a course with approximately 600 students, where 1/5 of grades are A's,
    1/4 are B's, 1/2 are C's, and the remaining are D's, the number of students who
    received a D is 30. -/
theorem students_with_d_grade (total_students : ℕ) (a_fraction b_fraction c_fraction : ℚ)
  (h_total : total_students = 600)
  (h_a : a_fraction = 1 / 5)
  (h_b : b_fraction = 1 / 4)
  (h_c : c_fraction = 1 / 2)
  (h_sum : a_fraction + b_fraction + c_fraction < 1) :
  total_students - (a_fraction + b_fraction + c_fraction) * total_students = 30 :=
sorry

end NUMINAMATH_CALUDE_students_with_d_grade_l1719_171984


namespace NUMINAMATH_CALUDE_shooting_competition_solution_l1719_171921

/-- Represents the number of shots for each score (8, 9, 10) -/
structure ScoreCounts where
  eight : ℕ
  nine : ℕ
  ten : ℕ

/-- Checks if a ScoreCounts satisfies the competition conditions -/
def is_valid_score (s : ScoreCounts) : Prop :=
  s.eight + s.nine + s.ten > 11 ∧
  8 * s.eight + 9 * s.nine + 10 * s.ten = 100

/-- The set of all valid score combinations -/
def valid_scores : Set ScoreCounts :=
  { s | is_valid_score s }

/-- The theorem stating the unique solution to the shooting competition problem -/
theorem shooting_competition_solution :
  valid_scores = { ⟨10, 0, 2⟩, ⟨9, 2, 1⟩, ⟨8, 4, 0⟩ } :=
sorry

end NUMINAMATH_CALUDE_shooting_competition_solution_l1719_171921


namespace NUMINAMATH_CALUDE_diamonds_15_diamonds_eq_diamonds_closed_diamonds_closed_15_l1719_171915

/-- The number of diamonds in the nth figure of the sequence -/
def diamonds (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else diamonds (n - 1) + 4 * n

/-- The theorem stating that the 15th figure contains 480 diamonds -/
theorem diamonds_15 : diamonds 15 = 480 := by
  sorry

/-- Alternative definition using the closed form formula -/
def diamonds_closed (n : ℕ) : ℕ := 2 * n * (n + 1)

/-- Theorem stating the equivalence of the recursive and closed form definitions -/
theorem diamonds_eq_diamonds_closed (n : ℕ) : diamonds n = diamonds_closed n := by
  sorry

/-- The theorem stating that the 15th figure contains 480 diamonds using the closed form -/
theorem diamonds_closed_15 : diamonds_closed 15 = 480 := by
  sorry

end NUMINAMATH_CALUDE_diamonds_15_diamonds_eq_diamonds_closed_diamonds_closed_15_l1719_171915


namespace NUMINAMATH_CALUDE_original_line_length_l1719_171998

/-- Proves that the original length of a line is 1 meter -/
theorem original_line_length
  (erased_length : ℝ)
  (remaining_length : ℝ)
  (h1 : erased_length = 33)
  (h2 : remaining_length = 67)
  (h3 : (100 : ℝ) = (1 : ℝ) * 100) :
  erased_length + remaining_length = 100 := by
sorry

end NUMINAMATH_CALUDE_original_line_length_l1719_171998


namespace NUMINAMATH_CALUDE_prism_no_circular_section_l1719_171917

/-- A solid object that can be cut by a plane -/
class Solid :=
  (can_produce_circular_section : Bool)

/-- A cone is a solid that can produce a circular cross-section -/
def Cone : Solid :=
  { can_produce_circular_section := true }

/-- A cylinder is a solid that can produce a circular cross-section -/
def Cylinder : Solid :=
  { can_produce_circular_section := true }

/-- A sphere is a solid that can produce a circular cross-section -/
def Sphere : Solid :=
  { can_produce_circular_section := true }

/-- A prism is a solid that cannot produce a circular cross-section -/
def Prism : Solid :=
  { can_produce_circular_section := false }

/-- Theorem: Among cones, cylinders, spheres, and prisms, only a prism cannot produce a circular cross-section -/
theorem prism_no_circular_section :
  ∀ s : Solid, s.can_produce_circular_section = false → s = Prism :=
by sorry

end NUMINAMATH_CALUDE_prism_no_circular_section_l1719_171917


namespace NUMINAMATH_CALUDE_sara_trout_count_l1719_171967

theorem sara_trout_count (melanie_trout : ℕ) (sara_trout : ℕ) : 
  melanie_trout = 10 → 
  melanie_trout = 2 * sara_trout → 
  sara_trout = 5 := by
sorry

end NUMINAMATH_CALUDE_sara_trout_count_l1719_171967


namespace NUMINAMATH_CALUDE_marco_marie_age_ratio_l1719_171979

theorem marco_marie_age_ratio :
  ∀ (x : ℕ) (marco_age marie_age : ℕ),
    marie_age = 12 →
    marco_age = x * marie_age + 1 →
    marco_age + marie_age = 37 →
    (marco_age : ℚ) / marie_age = 25 / 12 := by
  sorry

end NUMINAMATH_CALUDE_marco_marie_age_ratio_l1719_171979
