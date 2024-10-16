import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l134_13495

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l134_13495


namespace NUMINAMATH_CALUDE_alternate_multiply_divide_result_l134_13408

def alternateMultiplyDivide (n : ℕ) (initial : ℕ) : ℚ :=
  match n with
  | 0 => initial
  | m + 1 => if m % 2 = 0
             then (alternateMultiplyDivide m initial) * 3
             else (alternateMultiplyDivide m initial) / 2

theorem alternate_multiply_divide_result :
  alternateMultiplyDivide 15 (9^6) = 3^20 / 2^7 := by
  sorry

end NUMINAMATH_CALUDE_alternate_multiply_divide_result_l134_13408


namespace NUMINAMATH_CALUDE_cyclist_distance_difference_l134_13429

/-- The difference in distance traveled by two cyclists after five hours -/
theorem cyclist_distance_difference
  (daniel_distance : ℝ)
  (evan_initial_distance : ℝ)
  (evan_initial_time : ℝ)
  (evan_break_time : ℝ)
  (total_time : ℝ)
  (h1 : daniel_distance = 65)
  (h2 : evan_initial_distance = 40)
  (h3 : evan_initial_time = 3)
  (h4 : evan_break_time = 0.5)
  (h5 : total_time = 5) :
  daniel_distance - (evan_initial_distance + (evan_initial_distance / evan_initial_time) * (total_time - evan_initial_time - evan_break_time)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_distance_difference_l134_13429


namespace NUMINAMATH_CALUDE_tournament_rounds_theorem_l134_13483

/-- Represents a person in the tournament -/
inductive Person : Type
  | A
  | B
  | C

/-- Represents the tournament data -/
structure TournamentData where
  rounds_played : Person → Nat
  referee_rounds : Person → Nat

/-- The total number of rounds in the tournament -/
def total_rounds (data : TournamentData) : Nat :=
  (data.rounds_played Person.A + data.rounds_played Person.B + data.rounds_played Person.C + 
   data.referee_rounds Person.A + data.referee_rounds Person.B + data.referee_rounds Person.C) / 2

theorem tournament_rounds_theorem (data : TournamentData) 
  (h1 : data.rounds_played Person.A = 5)
  (h2 : data.rounds_played Person.B = 6)
  (h3 : data.referee_rounds Person.C = 2) :
  total_rounds data = 9 := by
  sorry

end NUMINAMATH_CALUDE_tournament_rounds_theorem_l134_13483


namespace NUMINAMATH_CALUDE_susan_third_turn_move_l134_13420

/-- A board game with the following properties:
  * The game board has 48 spaces from start to finish
  * A player moves 8 spaces forward on the first turn
  * On the second turn, the player moves 2 spaces forward but then 5 spaces backward
  * After the third turn, the player needs to move 37 more spaces to win
-/
structure BoardGame where
  total_spaces : Nat
  first_turn_move : Nat
  second_turn_forward : Nat
  second_turn_backward : Nat
  spaces_left_after_third_turn : Nat

/-- The specific game Susan is playing -/
def susans_game : BoardGame :=
  { total_spaces := 48
  , first_turn_move := 8
  , second_turn_forward := 2
  , second_turn_backward := 5
  , spaces_left_after_third_turn := 37 }

/-- Calculate the number of spaces moved on the third turn -/
def third_turn_move (game : BoardGame) : Nat :=
  game.total_spaces -
  (game.first_turn_move + game.second_turn_forward - game.second_turn_backward) -
  game.spaces_left_after_third_turn

/-- Theorem: Susan moved 6 spaces on the third turn -/
theorem susan_third_turn_move :
  third_turn_move susans_game = 6 := by
  sorry

end NUMINAMATH_CALUDE_susan_third_turn_move_l134_13420


namespace NUMINAMATH_CALUDE_shepherd_sheep_equations_correct_l134_13401

/-- Represents the number of sheep each shepherd has -/
structure ShepherdSheep where
  a : ℤ  -- number of sheep A has
  b : ℤ  -- number of sheep B has

/-- Checks if the given system of equations satisfies the conditions of the problem -/
def satisfies_conditions (s : ShepherdSheep) : Prop :=
  (s.a + 9 = 2 * (s.b - 9)) ∧ (s.b + 9 = s.a - 9)

/-- Theorem stating that the system of equations correctly represents the problem -/
theorem shepherd_sheep_equations_correct :
  ∃ (s : ShepherdSheep), satisfies_conditions s :=
sorry

end NUMINAMATH_CALUDE_shepherd_sheep_equations_correct_l134_13401


namespace NUMINAMATH_CALUDE_garden_length_l134_13427

/-- The length of a rectangular garden with perimeter 1800 m and breadth 400 m is 500 m. -/
theorem garden_length (perimeter breadth : ℝ) (h1 : perimeter = 1800) (h2 : breadth = 400) :
  (perimeter / 2 - breadth) = 500 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_l134_13427


namespace NUMINAMATH_CALUDE_polynomial_simplification_l134_13407

theorem polynomial_simplification (w : ℝ) : 
  3*w + 4 - 2*w^2 - 5*w - 6 + w^2 + 7*w + 8 - 3*w^2 = 5*w - 4*w^2 + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l134_13407


namespace NUMINAMATH_CALUDE_binomial_10_2_l134_13404

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

/-- Theorem: The binomial coefficient (10 choose 2) equals 45 -/
theorem binomial_10_2 : binomial 10 2 = 45 := by sorry

end NUMINAMATH_CALUDE_binomial_10_2_l134_13404


namespace NUMINAMATH_CALUDE_inverse_composition_result_l134_13418

-- Define the functions f and h
variable (f h : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv h_inv : ℝ → ℝ)

-- State the given condition
axiom condition : ∀ x, f_inv (h x) = 6 * x - 4

-- State the theorem to be proved
theorem inverse_composition_result : h_inv (f 3) = 7/6 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_result_l134_13418


namespace NUMINAMATH_CALUDE_x_squared_is_quadratic_l134_13425

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² = 0 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem stating that x² = 0 is a quadratic equation -/
theorem x_squared_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_is_quadratic_l134_13425


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_average_l134_13494

theorem quadratic_equation_roots_average (a b : ℝ) (h : a ≠ 0) : 
  let f : ℝ → ℝ := λ x ↦ 3*a*x^2 - 6*a*x + 2*b
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) → (x₁ + x₂) / 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_average_l134_13494


namespace NUMINAMATH_CALUDE_total_candies_l134_13449

/-- The total number of candies for six people given specific relationships between their candy counts. -/
theorem total_candies (adam james rubert lisa chris emily : ℕ) : 
  adam = 6 ∧ 
  james = 3 * adam ∧ 
  rubert = 4 * james ∧ 
  lisa = 2 * rubert ∧ 
  chris = lisa + 5 ∧ 
  emily = 3 * chris - 7 → 
  adam + james + rubert + lisa + chris + emily = 829 := by
  sorry

#eval 6 + 3 * 6 + 4 * (3 * 6) + 2 * (4 * (3 * 6)) + (2 * (4 * (3 * 6)) + 5) + (3 * (2 * (4 * (3 * 6)) + 5) - 7)

end NUMINAMATH_CALUDE_total_candies_l134_13449


namespace NUMINAMATH_CALUDE_ellipse_and_fixed_point_l134_13410

noncomputable section

/-- The ellipse C with given conditions -/
structure Ellipse :=
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : (c / b) = Real.sqrt 3 / 3)
  (h5 : b + c + 2*c = 3 + Real.sqrt 3)

/-- The equation of the ellipse is x²/4 + y²/3 = 1 -/
def ellipse_equation (C : Ellipse) : Prop :=
  C.a = 2 ∧ C.b = Real.sqrt 3

/-- The fixed point on x-axis -/
def fixed_point : ℝ × ℝ := (5/2, 0)

/-- The line QM passes through the fixed point -/
def line_passes_through_fixed_point (C : Ellipse) (P : ℝ × ℝ) : Prop :=
  let F := (C.c, 0)
  let M := (4, P.2)
  let Q := sorry -- Intersection of PF with the ellipse
  ∃ t : ℝ, fixed_point = (1 - t) • Q + t • M

/-- Main theorem -/
theorem ellipse_and_fixed_point (C : Ellipse) :
  ellipse_equation C ∧
  ∀ P, P.1^2 / 4 + P.2^2 / 3 = 1 → line_passes_through_fixed_point C P :=
sorry

end

end NUMINAMATH_CALUDE_ellipse_and_fixed_point_l134_13410


namespace NUMINAMATH_CALUDE_units_digit_fourth_power_not_seven_l134_13469

theorem units_digit_fourth_power_not_seven :
  ∀ n : ℕ, (n^4 % 10) ≠ 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_fourth_power_not_seven_l134_13469


namespace NUMINAMATH_CALUDE_inequality_proof_l134_13460

theorem inequality_proof (a b c : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0)
  (sum_one : a + b + c = 1) : 
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l134_13460


namespace NUMINAMATH_CALUDE_marble_selection_ways_l134_13421

def total_marbles : ℕ := 15
def specific_colors : ℕ := 5
def marbles_to_choose : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem marble_selection_ways :
  (specific_colors * choose (total_marbles - specific_colors - 1) (marbles_to_choose - 1)) = 630 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l134_13421


namespace NUMINAMATH_CALUDE_permutation_sum_consecutive_l134_13446

theorem permutation_sum_consecutive (n : ℕ) (h : n ≥ 2) :
  (∃ (a b : Fin n → Fin n),
    Function.Bijective a ∧ Function.Bijective b ∧
    ∃ (k : ℕ), ∀ i : Fin n, (a i).val + (b i).val = k + i.val) ↔
  Odd n :=
sorry

end NUMINAMATH_CALUDE_permutation_sum_consecutive_l134_13446


namespace NUMINAMATH_CALUDE_orange_theft_ratio_l134_13448

/-- Proves the ratio of stolen oranges to remaining oranges is 1:2 --/
theorem orange_theft_ratio :
  ∀ (initial_oranges eaten_oranges returned_oranges final_oranges : ℕ),
    initial_oranges = 60 →
    eaten_oranges = 10 →
    returned_oranges = 5 →
    final_oranges = 30 →
    ∃ (stolen_oranges : ℕ),
      stolen_oranges = initial_oranges - eaten_oranges - (final_oranges - returned_oranges) ∧
      2 * stolen_oranges = initial_oranges - eaten_oranges :=
by
  sorry

#check orange_theft_ratio

end NUMINAMATH_CALUDE_orange_theft_ratio_l134_13448


namespace NUMINAMATH_CALUDE_min_value_of_function_l134_13403

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  x + 2 / (2 * x + 1) - 1 ≥ 1 / 2 ∧ 
  ∃ y > 0, y + 2 / (2 * y + 1) - 1 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l134_13403


namespace NUMINAMATH_CALUDE_even_odd_sum_l134_13496

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

/-- A function g: ℝ → ℝ is odd if g(-x) = -g(x) for all x ∈ ℝ -/
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x

/-- Given f and g are even and odd functions respectively, and f(x) - g(x) = x^3 + x^2 + 1,
    prove that f(1) + g(1) = 1 -/
theorem even_odd_sum (f g : ℝ → ℝ) (hf : IsEven f) (hg : IsOdd g)
    (h : ∀ x : ℝ, f x - g x = x^3 + x^2 + 1) : f 1 + g 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_l134_13496


namespace NUMINAMATH_CALUDE_square_side_irrational_l134_13499

theorem square_side_irrational (area : ℝ) (h : area = 3) :
  ∃ (side : ℝ), side * side = area ∧ Irrational side := by
  sorry

end NUMINAMATH_CALUDE_square_side_irrational_l134_13499


namespace NUMINAMATH_CALUDE_min_value_of_m_l134_13419

theorem min_value_of_m (x y : ℝ) (h1 : y = x^2 - 2) (h2 : x > Real.sqrt 3) :
  let m := (3*x + y - 4)/(x - 1) + (x + 3*y - 4)/(y - 1)
  m ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), y₀ = x₀^2 - 2 ∧ x₀ > Real.sqrt 3 ∧
    (3*x₀ + y₀ - 4)/(x₀ - 1) + (x₀ + 3*y₀ - 4)/(y₀ - 1) = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_m_l134_13419


namespace NUMINAMATH_CALUDE_square_roots_problem_l134_13435

theorem square_roots_problem (m a : ℝ) (hm : m > 0) 
  (h1 : (1 - 2*a)^2 = m) (h2 : (a - 5)^2 = m) (h3 : 1 - 2*a ≠ a - 5) : 
  m = 81 := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l134_13435


namespace NUMINAMATH_CALUDE_reflection_line_sum_l134_13498

/-- Given a reflection of point (2, -2) across line y = mx + b to point (8, 4), prove m + b = 5 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The reflected point (x, y) satisfies the reflection property
    (x - 2)^2 + (y + 2)^2 = (8 - 2)^2 + (4 + 2)^2 ∧
    -- The midpoint of the original and reflected points lies on y = mx + b
    (1 : ℝ) = m * 5 + b ∧
    -- The line y = mx + b is perpendicular to the line connecting the original and reflected points
    m * ((8 - 2) / (4 + 2)) = -1) →
  m + b = 5 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l134_13498


namespace NUMINAMATH_CALUDE_problem_solution_l134_13467

theorem problem_solution (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < 0) (h4 : 0 < c) :
  (a * b < b * c) ∧ (a * c < b * c) ∧ (a + b < b + c) ∧ (c / a < 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l134_13467


namespace NUMINAMATH_CALUDE_find_N_l134_13453

/-- Given three numbers a, b, and c, and a value N, satisfying certain conditions,
    prove that N = 41 is the integer solution that best satisfies all conditions. -/
theorem find_N : ∃ (a b c N : ℚ),
  a + b + c = 90 ∧
  a - 7 = N ∧
  b + 7 = N ∧
  5 * c = N ∧
  N.floor = 41 :=
by sorry

end NUMINAMATH_CALUDE_find_N_l134_13453


namespace NUMINAMATH_CALUDE_minimum_students_for_photo_l134_13470

def photo_cost (x : ℝ) : ℝ := 5 + (x - 2) * 0.8

theorem minimum_students_for_photo : 
  ∃ x : ℝ, x ≥ 17 ∧ 
  (∀ y : ℝ, y ≥ x → photo_cost y / y ≤ 1) ∧
  (∀ z : ℝ, z < x → photo_cost z / z > 1) :=
sorry

end NUMINAMATH_CALUDE_minimum_students_for_photo_l134_13470


namespace NUMINAMATH_CALUDE_gravitational_force_in_orbit_l134_13481

/-- Gravitational force calculation -/
theorem gravitational_force_in_orbit 
  (surface_distance : ℝ) 
  (orbit_distance : ℝ) 
  (surface_force : ℝ) 
  (h1 : surface_distance = 6000)
  (h2 : orbit_distance = 36000)
  (h3 : surface_force = 800)
  (h4 : ∀ (d f : ℝ), f * d^2 = surface_force * surface_distance^2) :
  ∃ (orbit_force : ℝ), 
    orbit_force * orbit_distance^2 = surface_force * surface_distance^2 ∧ 
    orbit_force = 1 / 45 := by
  sorry

end NUMINAMATH_CALUDE_gravitational_force_in_orbit_l134_13481


namespace NUMINAMATH_CALUDE_is_factorization_l134_13441

/-- Proves that x^2 - 4x + 4 = (x - 2)^2 is a factorization --/
theorem is_factorization (x : ℝ) : x^2 - 4*x + 4 = (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_is_factorization_l134_13441


namespace NUMINAMATH_CALUDE_max_servings_is_eight_l134_13416

/-- Represents the recipe requirements for 4 servings --/
structure Recipe :=
  (eggs : ℚ)
  (sugar : ℚ)
  (milk : ℚ)

/-- Represents Lisa's available ingredients --/
structure Available :=
  (eggs : ℚ)
  (sugar : ℚ)
  (milk : ℚ)

/-- Calculates the maximum number of servings possible for a given ingredient --/
def max_servings_for_ingredient (recipe_amount : ℚ) (available_amount : ℚ) : ℚ :=
  (available_amount / recipe_amount) * 4

/-- Finds the maximum number of servings possible given the recipe and available ingredients --/
def max_servings (recipe : Recipe) (available : Available) : ℚ :=
  min (max_servings_for_ingredient recipe.eggs available.eggs)
    (min (max_servings_for_ingredient recipe.sugar available.sugar)
      (max_servings_for_ingredient recipe.milk available.milk))

theorem max_servings_is_eight :
  let recipe := Recipe.mk 3 (1/2) 2
  let available := Available.mk 10 1 9
  max_servings recipe available = 8 := by
  sorry

#eval max_servings (Recipe.mk 3 (1/2) 2) (Available.mk 10 1 9)

end NUMINAMATH_CALUDE_max_servings_is_eight_l134_13416


namespace NUMINAMATH_CALUDE_markup_rate_l134_13457

theorem markup_rate (S : ℝ) (h1 : S > 0) : 
  let profit := 0.20 * S
  let expenses := 0.20 * S
  let cost := S - profit - expenses
  (S - cost) / cost * 100 = 200 / 3 := by sorry

end NUMINAMATH_CALUDE_markup_rate_l134_13457


namespace NUMINAMATH_CALUDE_min_triangles_17gon_is_six_l134_13473

/-- The minimum number of triangles needed to divide a 17-gon -/
def min_triangles_17gon : ℕ := 6

/-- A polygon with 17 sides -/
structure Polygon17 :=
  (vertices : Fin 17 → ℝ × ℝ)

/-- A triangulation of a polygon -/
structure Triangulation (P : Polygon17) :=
  (num_triangles : ℕ)
  (is_valid : num_triangles ≥ min_triangles_17gon)

/-- Theorem: The minimum number of triangles to divide a 17-gon is 6 -/
theorem min_triangles_17gon_is_six (P : Polygon17) :
  ∀ (T : Triangulation P), T.num_triangles ≥ min_triangles_17gon :=
sorry

end NUMINAMATH_CALUDE_min_triangles_17gon_is_six_l134_13473


namespace NUMINAMATH_CALUDE_player_a_wins_l134_13440

/-- Represents a point on the coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a move in the game -/
inductive Move
  | right : Move
  | up : Move

/-- Represents the game state -/
structure GameState where
  piecePosition : Point
  markedPoints : Set Point
  movesLeft : ℕ

/-- The game rules -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  match move with
  | Move.right => 
    let newPos := Point.mk (state.piecePosition.x + 1) state.piecePosition.y
    newPos ∉ state.markedPoints
  | Move.up => 
    let newPos := Point.mk state.piecePosition.x (state.piecePosition.y + 1)
    newPos ∉ state.markedPoints

/-- Player A's strategy -/
def strategyA (k : ℕ) (state : GameState) : Point := sorry

/-- Theorem: Player A has a winning strategy for any positive k -/
theorem player_a_wins (k : ℕ) (h : k > 0) : 
  ∃ (strategy : GameState → Point), 
    ∀ (initialState : GameState),
      (∀ (move : Move), ¬isValidMove initialState move) ∨
      (∃ (finalState : GameState), 
        finalState.markedPoints = insert (strategy initialState) initialState.markedPoints ∧
        ∀ (move : Move), ¬isValidMove finalState move) := by
  sorry

end NUMINAMATH_CALUDE_player_a_wins_l134_13440


namespace NUMINAMATH_CALUDE_pizza_order_l134_13458

theorem pizza_order (people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) 
  (h1 : people = 18) 
  (h2 : slices_per_person = 3) 
  (h3 : slices_per_pizza = 9) : 
  (people * slices_per_person) / slices_per_pizza = 6 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_l134_13458


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l134_13411

/-- The line 4x + 7y + 49 = 0 is tangent to the parabola y^2 = 16x -/
theorem line_tangent_to_parabola :
  ∃! (x y : ℝ), 4 * x + 7 * y + 49 = 0 ∧ y^2 = 16 * x := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l134_13411


namespace NUMINAMATH_CALUDE_derivative_not_critical_point_l134_13422

-- Define the function g(x) as the derivative of f(x)
def g (a : ℝ) (x : ℝ) : ℝ := (x - 1) * (x^2 - 3*x + a)

-- State the theorem
theorem derivative_not_critical_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, (deriv f) x = g a x) →  -- The derivative of f is g
  (deriv f) 1 ≠ 0 →             -- 1 is not a critical point
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_derivative_not_critical_point_l134_13422


namespace NUMINAMATH_CALUDE_percentage_of_girls_l134_13491

theorem percentage_of_girls (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 900 →
  boys = 90 →
  girls = total - boys →
  (girls : ℚ) / (total : ℚ) * 100 = 90 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_girls_l134_13491


namespace NUMINAMATH_CALUDE_point_labeling_theorem_l134_13428

/-- A point in the space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- The set of n points in the space -/
def PointSet (n : ℕ) := Fin n → Point

theorem point_labeling_theorem (n : ℕ) (points : PointSet n) 
  (h : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → 
    ∃ (p : Fin 3 → Fin n), angle (points (p 0)) (points (p 1)) (points (p 2)) > 120) :
  ∃ (σ : Equiv (Fin n) (Fin n)), 
    ∀ (i j k : Fin n), i < j → j < k → 
      angle (points (σ i)) (points (σ j)) (points (σ k)) > 120 :=
sorry

end NUMINAMATH_CALUDE_point_labeling_theorem_l134_13428


namespace NUMINAMATH_CALUDE_point_symmetry_wrt_origin_l134_13454

/-- Given a point M with coordinates (-2,3), its coordinates with respect to the origin are (2,-3). -/
theorem point_symmetry_wrt_origin : 
  let M : ℝ × ℝ := (-2, 3)
  (- M.1, - M.2) = (2, -3) := by sorry

end NUMINAMATH_CALUDE_point_symmetry_wrt_origin_l134_13454


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l134_13402

/-- The measure of each interior angle in a regular octagon -/
theorem regular_octagon_interior_angle : ℝ := by
  -- Define the number of sides in an octagon
  let n : ℕ := 8

  -- Define the formula for the sum of interior angles of an n-sided polygon
  let sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

  -- Define the measure of each interior angle in a regular n-gon
  let interior_angle (n : ℕ) : ℝ := sum_interior_angles n / n

  -- Prove that the measure of each interior angle in a regular octagon is 135°
  have h : interior_angle n = 135 := by sorry

  -- Return the result
  exact 135


end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l134_13402


namespace NUMINAMATH_CALUDE_min_value_theorem_l134_13464

/-- Given a function y = a^x + b where b > 0, a > 1, and 3 = a + b, 
    the minimum value of (4 / (a - 1)) + (1 / b) is 9/2 -/
theorem min_value_theorem (a b : ℝ) (h1 : b > 0) (h2 : a > 1) (h3 : 3 = a + b) :
  (∀ x : ℝ, (4 / (a - 1)) + (1 / b) ≥ 9/2) ∧ 
  (∃ x : ℝ, (4 / (a - 1)) + (1 / b) = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l134_13464


namespace NUMINAMATH_CALUDE_paco_sweet_cookies_remaining_l134_13474

/-- Given the initial number of sweet cookies and the number eaten, 
    calculate the number of sweet cookies remaining. -/
def sweet_cookies_remaining (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem stating that for Paco's specific case, 
    the number of sweet cookies remaining is 19. -/
theorem paco_sweet_cookies_remaining : 
  sweet_cookies_remaining 34 15 = 19 := by
  sorry

end NUMINAMATH_CALUDE_paco_sweet_cookies_remaining_l134_13474


namespace NUMINAMATH_CALUDE_extremum_at_negative_three_l134_13480

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

-- Theorem statement
theorem extremum_at_negative_three (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -3 ∧ |x + 3| < ε → f a x ≤ f a (-3)) →
  a = 5 :=
sorry

end NUMINAMATH_CALUDE_extremum_at_negative_three_l134_13480


namespace NUMINAMATH_CALUDE_greatest_common_length_l134_13413

theorem greatest_common_length (a b c d : ℕ) 
  (ha : a = 48) (hb : b = 60) (hc : c = 72) (hd : d = 120) : 
  Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_length_l134_13413


namespace NUMINAMATH_CALUDE_max_value_log_sum_l134_13472

theorem max_value_log_sum (a b : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : a * b = 1000) :
  Real.sqrt (1 + Real.log a / Real.log 10) + Real.sqrt (1 + Real.log b / Real.log 10) ≤ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_log_sum_l134_13472


namespace NUMINAMATH_CALUDE_complex_equation_solution_l134_13462

theorem complex_equation_solution (z : ℂ) (b : ℝ) :
  z * (1 + Complex.I) = 1 - b * Complex.I →
  Complex.abs z = Real.sqrt 2 →
  b = Real.sqrt 3 ∨ b = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l134_13462


namespace NUMINAMATH_CALUDE_hexagon_triangle_area_l134_13476

/-- The area of an equilateral triangle formed by connecting the second, third, and fifth vertices
    of a regular hexagon with side length 12 cm is 36√3 cm^2. -/
theorem hexagon_triangle_area :
  let hexagon_side : ℝ := 12
  let triangle_side : ℝ := hexagon_side
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side ^ 2
  triangle_area = 36 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hexagon_triangle_area_l134_13476


namespace NUMINAMATH_CALUDE_lcm_12_20_l134_13477

theorem lcm_12_20 : Nat.lcm 12 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_20_l134_13477


namespace NUMINAMATH_CALUDE_exactly_one_success_probability_l134_13433

theorem exactly_one_success_probability (p : ℝ) (h1 : p = 1/3) : 
  3 * (1 - p) * p^2 = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_success_probability_l134_13433


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l134_13484

-- Define binary numbers as natural numbers
def bin1010 : ℕ := 10
def bin111 : ℕ := 7
def bin1001 : ℕ := 9
def bin1011 : ℕ := 11
def bin10111 : ℕ := 23

-- Theorem statement
theorem binary_arithmetic_equality :
  (bin1010 + bin111) - bin1001 + bin1011 = bin10111 := by
  sorry

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l134_13484


namespace NUMINAMATH_CALUDE_dinner_lunch_ratio_l134_13487

/-- Represents the amount of bread eaten at each meal in grams -/
structure BreadConsumption where
  breakfast : ℕ
  lunch : ℕ
  dinner : ℕ

/-- Proves that given the conditions, the ratio of dinner bread to lunch bread is 8:1 -/
theorem dinner_lunch_ratio (b : BreadConsumption) : 
  b.dinner = 240 ∧ 
  ∃ k : ℕ, b.dinner = k * b.lunch ∧ 
  b.dinner = 6 * b.breakfast ∧ 
  b.breakfast + b.lunch + b.dinner = 310 → 
  b.dinner / b.lunch = 8 := by
  sorry

end NUMINAMATH_CALUDE_dinner_lunch_ratio_l134_13487


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_values_l134_13438

theorem fraction_zero_implies_x_values (x : ℝ) : 
  (x ^ 2 - 4) / x = 0 → x = 2 ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_values_l134_13438


namespace NUMINAMATH_CALUDE_burn_represents_8615_l134_13459

/-- Represents a mapping from characters to digits -/
def DigitMapping := Char → Fin 10

/-- The sequence of characters used in the code -/
def codeSequence : List Char := ['G', 'R', 'E', 'A', 'T', 'N', 'U', 'M', 'B', 'S']

/-- Creates a mapping from the code sequence to digits 0-9 -/
def createMapping (seq : List Char) : DigitMapping :=
  fun c => match seq.indexOf? c with
    | some i => ⟨i, by sorry⟩
    | none => 0

/-- The mapping for our specific code -/
def mapping : DigitMapping := createMapping codeSequence

/-- Converts a string to a number using the given mapping -/
def stringToNumber (s : String) (m : DigitMapping) : Nat :=
  s.foldr (fun c acc => acc * 10 + m c) 0

theorem burn_represents_8615 :
  stringToNumber "BURN" mapping = 8615 := by sorry

end NUMINAMATH_CALUDE_burn_represents_8615_l134_13459


namespace NUMINAMATH_CALUDE_parabola_perpendicular_chords_theorem_l134_13406

/-- A parabola with vertex at the origin and focus on the positive x-axis -/
structure Parabola where
  p : ℝ
  equation : ℝ × ℝ → Prop := fun (x, y) ↦ y^2 = 2 * p * x

/-- A line passing through two points -/
def Line (A B : ℝ × ℝ) : ℝ × ℝ → Prop :=
  fun P ↦ (P.2 - A.2) * (B.1 - A.1) = (P.1 - A.1) * (B.2 - A.2)

/-- Two lines are perpendicular -/
def Perpendicular (L₁ L₂ : (ℝ × ℝ → Prop)) : Prop :=
  ∃ A B C D, L₁ A ∧ L₁ B ∧ L₂ C ∧ L₂ D ∧
    (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0

/-- The projection of a point onto a line -/
def Projection (P : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ × ℝ → Prop :=
  fun H ↦ L H ∧ Perpendicular (Line P H) L

theorem parabola_perpendicular_chords_theorem (Γ : Parabola) :
  ∀ A B, Γ.equation A ∧ Γ.equation B ∧ 
         Perpendicular (Line (0, 0) A) (Line (0, 0) B) →
  (∃ M₀, M₀ = (2 * Γ.p, 0) ∧ Line A B M₀) ∧
  (∀ H, Projection (0, 0) (Line A B) H → 
        H.1^2 + H.2^2 - 2 * Γ.p * H.1 = 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_perpendicular_chords_theorem_l134_13406


namespace NUMINAMATH_CALUDE_root_product_plus_one_l134_13409

theorem root_product_plus_one (p q r : ℂ) : 
  p^3 - 15*p^2 + 10*p + 24 = 0 →
  q^3 - 15*q^2 + 10*q + 24 = 0 →
  r^3 - 15*r^2 + 10*r + 24 = 0 →
  (1+p)*(1+q)*(1+r) = 2 := by
sorry

end NUMINAMATH_CALUDE_root_product_plus_one_l134_13409


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l134_13490

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set Nat := {2, 4, 5}

-- Define set B
def B : Set Nat := {x ∈ U | 2 < x ∧ x < 6}

-- Theorem statement
theorem complement_A_intersect_B : 
  (U \ A) ∩ B = {3} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l134_13490


namespace NUMINAMATH_CALUDE_no_alpha_exists_for_inequality_l134_13475

theorem no_alpha_exists_for_inequality :
  ∀ α : ℝ, α > 0 → ∃ x : ℝ, |Real.cos x| + |Real.cos (α * x)| ≤ Real.sin x + Real.sin (α * x) := by
  sorry

end NUMINAMATH_CALUDE_no_alpha_exists_for_inequality_l134_13475


namespace NUMINAMATH_CALUDE_linear_function_proof_l134_13414

/-- A linear function passing through (-2, -1) and parallel to y = 2x - 3 -/
def f (x : ℝ) : ℝ := 2 * x + 3

/-- The slope of the line y = 2x - 3 -/
def slope_parallel : ℝ := 2

theorem linear_function_proof :
  (∀ x, f x = 2 * x + 3) ∧
  f (-2) = -1 ∧
  (∀ x y, f y - f x = slope_parallel * (y - x)) :=
sorry

end NUMINAMATH_CALUDE_linear_function_proof_l134_13414


namespace NUMINAMATH_CALUDE_tyson_total_score_l134_13415

/-- The number of times Tyson scored three points -/
def three_pointers : ℕ := 15

/-- The number of times Tyson scored two points -/
def two_pointers : ℕ := 12

/-- The number of times Tyson scored one point -/
def one_pointers : ℕ := 6

/-- The total points Tyson scored -/
def total_points : ℕ := 3 * three_pointers + 2 * two_pointers + one_pointers

theorem tyson_total_score : total_points = 75 := by
  sorry

end NUMINAMATH_CALUDE_tyson_total_score_l134_13415


namespace NUMINAMATH_CALUDE_constant_function_theorem_l134_13445

/-- A function f: ℝ → ℝ is twice differentiable if it has a second derivative -/
def TwiceDifferentiable (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ Differentiable ℝ (deriv f)

/-- The given inequality condition for the function f -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (deriv^[2] f x) * Real.cos (f x) ≥ (deriv f x)^2 * Real.sin (f x)

/-- Main theorem: If f is twice differentiable and satisfies the inequality,
    then f is a constant function -/
theorem constant_function_theorem (f : ℝ → ℝ) 
    (h1 : TwiceDifferentiable f) (h2 : SatisfiesInequality f) :
    ∃ k : ℝ, ∀ x : ℝ, f x = k := by
  sorry

end NUMINAMATH_CALUDE_constant_function_theorem_l134_13445


namespace NUMINAMATH_CALUDE_norm_took_110_photos_l134_13497

/-- The number of photos taken by Norm given the conditions of the problem -/
def norm_photos (lisa mike norm : ℕ) : Prop :=
  (lisa + mike = mike + norm - 60) ∧ 
  (norm = 2 * lisa + 10) ∧
  (norm = 110)

/-- Theorem stating that Norm took 110 photos given the problem conditions -/
theorem norm_took_110_photos :
  ∃ (lisa mike norm : ℕ), norm_photos lisa mike norm :=
by
  sorry

end NUMINAMATH_CALUDE_norm_took_110_photos_l134_13497


namespace NUMINAMATH_CALUDE_aluminum_cans_collection_l134_13479

theorem aluminum_cans_collection : 
  let sarah_yesterday : ℕ := 50
  let lara_yesterday : ℕ := sarah_yesterday + 30
  let sarah_today : ℕ := 40
  let lara_today : ℕ := 70
  let total_yesterday : ℕ := sarah_yesterday + lara_yesterday
  let total_today : ℕ := sarah_today + lara_today
  total_yesterday - total_today = 20 := by
sorry

end NUMINAMATH_CALUDE_aluminum_cans_collection_l134_13479


namespace NUMINAMATH_CALUDE_polynomial_factor_l134_13452

theorem polynomial_factor (a : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 + a*x - 5 = (x - 2) * (x + k)) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l134_13452


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_nine_l134_13485

theorem gcd_factorial_eight_nine : Nat.gcd (Nat.factorial 8) (Nat.factorial 9) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_nine_l134_13485


namespace NUMINAMATH_CALUDE_discount_percentage_l134_13471

theorem discount_percentage (original_price sale_price : ℝ) 
  (h1 : original_price = 150)
  (h2 : sale_price = 135) :
  (original_price - sale_price) / original_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l134_13471


namespace NUMINAMATH_CALUDE_largest_angle_convex_pentagon_l134_13466

theorem largest_angle_convex_pentagon (x : ℚ) :
  (x + 2) + (2*x + 3) + (3*x + 6) + (4*x + 5) + (5*x + 4) = 540 →
  max (x + 2) (max (2*x + 3) (max (3*x + 6) (max (4*x + 5) (5*x + 4)))) = 532 / 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_convex_pentagon_l134_13466


namespace NUMINAMATH_CALUDE_package_weight_problem_l134_13432

theorem package_weight_problem (a b c : ℝ) 
  (hab : a + b = 108)
  (hbc : b + c = 132)
  (hca : c + a = 138) :
  a + b + c = 189 ∧ a ≥ 40 ∧ b ≥ 40 ∧ c ≥ 40 := by
  sorry

end NUMINAMATH_CALUDE_package_weight_problem_l134_13432


namespace NUMINAMATH_CALUDE_min_value_theorem_l134_13465

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 9*x + 81/x^4 ≥ 19 ∧
  (x^2 + 9*x + 81/x^4 = 19 ↔ x = 3) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l134_13465


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l134_13442

theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (1 / m + 1 / n) ≥ 4 ∧ ∃ m n, m > 0 ∧ n > 0 ∧ m + n = 1 ∧ 1 / m + 1 / n = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l134_13442


namespace NUMINAMATH_CALUDE_inequality_proof_l134_13434

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) : 
  Real.sqrt (b^2 - a*c) < Real.sqrt (3*a) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l134_13434


namespace NUMINAMATH_CALUDE_monkey_count_l134_13424

/-- Given a group of monkeys that can eat 6 bananas in 6 minutes and 18 bananas in 18 minutes,
    prove that there are 6 monkeys in the group. -/
theorem monkey_count (eating_rate : ℕ → ℕ → ℕ) (monkey_count : ℕ) : 
  (eating_rate 6 6 = 6) →  -- 6 bananas in 6 minutes
  (eating_rate 18 18 = 18) →  -- 18 bananas in 18 minutes
  monkey_count = 6 :=
by sorry

end NUMINAMATH_CALUDE_monkey_count_l134_13424


namespace NUMINAMATH_CALUDE_rotation_sum_65_l134_13447

/-- Triangle in 2D space defined by three points -/
structure Triangle where
  x : ℝ × ℝ
  y : ℝ × ℝ
  z : ℝ × ℝ

/-- Rotation in 2D space defined by an angle and a center point -/
structure Rotation where
  angle : ℝ
  center : ℝ × ℝ

/-- Check if two triangles are congruent under rotation -/
def isCongruentUnderRotation (t1 t2 : Triangle) (r : Rotation) : Prop :=
  sorry

theorem rotation_sum_65 (xyz x'y'z' : Triangle) (r : Rotation) :
  xyz.x = (0, 0) →
  xyz.y = (0, 15) →
  xyz.z = (20, 0) →
  x'y'z'.x = (30, 10) →
  x'y'z'.y = (40, 10) →
  x'y'z'.z = (30, 0) →
  isCongruentUnderRotation xyz x'y'z' r →
  r.angle ≤ r'.angle → isCongruentUnderRotation xyz x'y'z' r' →
  r.angle + r.center.1 + r.center.2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_rotation_sum_65_l134_13447


namespace NUMINAMATH_CALUDE_rectangle_circle_tangent_l134_13437

/-- Given a circle with radius 6 cm tangent to two shorter sides and one longer side of a rectangle,
    and the area of the rectangle being three times the area of the circle,
    prove that the length of the shorter side of the rectangle is 12 cm. -/
theorem rectangle_circle_tangent (circle_radius : ℝ) (rectangle_area : ℝ) (circle_area : ℝ) :
  circle_radius = 6 →
  rectangle_area = 3 * circle_area →
  circle_area = Real.pi * circle_radius^2 →
  (12 : ℝ) = 2 * circle_radius :=
by sorry

end NUMINAMATH_CALUDE_rectangle_circle_tangent_l134_13437


namespace NUMINAMATH_CALUDE_triangle_area_qin_jiushao_l134_13492

theorem triangle_area_qin_jiushao 
  (a b c : ℝ) 
  (h_positive : 0 < c ∧ 0 < b ∧ 0 < a) 
  (h_order : c < b ∧ b < a) 
  (h_a : a = 15) 
  (h_b : b = 14) 
  (h_c : c = 13) : 
  Real.sqrt ((1/4) * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2)) = 84 := by
  sorry

#check triangle_area_qin_jiushao

end NUMINAMATH_CALUDE_triangle_area_qin_jiushao_l134_13492


namespace NUMINAMATH_CALUDE_pizza_slices_sold_l134_13468

/-- Proves that the number of small slices sold is 2000 -/
theorem pizza_slices_sold (small_price large_price : ℕ) 
  (total_slices total_revenue : ℕ) (h1 : small_price = 150) 
  (h2 : large_price = 250) (h3 : total_slices = 5000) 
  (h4 : total_revenue = 1050000) : 
  ∃ (small_slices large_slices : ℕ),
    small_slices + large_slices = total_slices ∧
    small_price * small_slices + large_price * large_slices = total_revenue ∧
    small_slices = 2000 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_sold_l134_13468


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l134_13450

theorem circle_center_radius_sum :
  ∀ (a b r : ℝ),
  (∀ (x y : ℝ), x^2 + 16*y + 97 = -y^2 - 8*x ↔ (x - a)^2 + (y - b)^2 = r^2) →
  a + b + r = -12 + Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l134_13450


namespace NUMINAMATH_CALUDE_gcd_lcm_problem_l134_13493

theorem gcd_lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 24) (h2 : Nat.lcm a b = 3600) (h3 : b = 240) : a = 360 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_problem_l134_13493


namespace NUMINAMATH_CALUDE_counterexample_exists_l134_13488

theorem counterexample_exists : ∃ n : ℕ, ¬ Nat.Prime n ∧ ¬ Nat.Prime (n - 3) ∧ n = 18 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l134_13488


namespace NUMINAMATH_CALUDE_max_quarters_proof_l134_13482

/-- Represents the number of coins of each type --/
structure CoinCount where
  quarters : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value of coins in cents --/
def totalValue (coins : CoinCount) : ℕ :=
  coins.quarters * 25 + coins.nickels * 5 + coins.dimes * 10

/-- Checks if the coin count satisfies the problem conditions --/
def isValidCount (coins : CoinCount) : Prop :=
  coins.quarters = coins.nickels ∧ coins.dimes * 2 = coins.quarters

/-- The maximum number of quarters possible given the conditions --/
def maxQuarters : ℕ := 11

theorem max_quarters_proof :
  ∀ coins : CoinCount,
    isValidCount coins →
    totalValue coins = 400 →
    coins.quarters ≤ maxQuarters :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_proof_l134_13482


namespace NUMINAMATH_CALUDE_function_inequality_l134_13489

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of f
variable (h : ∀ x, x > 0 → DifferentiableAt ℝ f x)

-- Define the condition f(x)/x > f'(x)
variable (cond : ∀ x, x > 0 → (f x) / x > deriv f x)

-- Theorem statement
theorem function_inequality : 2015 * (f 2016) > 2016 * (f 2015) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l134_13489


namespace NUMINAMATH_CALUDE_centroid_trace_area_l134_13455

-- Define the circle
def Circle : Type := {p : ℝ × ℝ // (p.1^2 + p.2^2 = 225)}

-- Define points A, B, and C
def A : ℝ × ℝ := (-15, 0)
def B : ℝ × ℝ := (15, 0)

-- Define C as a point on the circle
def C : Circle := sorry

-- Define the centroid of triangle ABC
def centroid (c : Circle) : ℝ × ℝ := sorry

-- Statement to prove
theorem centroid_trace_area :
  ∃ (area : ℝ), area = 25 * Real.pi ∧
  (∀ (c : Circle), c.1 ≠ A ∧ c.1 ≠ B →
    (centroid c).1^2 + (centroid c).2^2 = 25) :=
sorry

end NUMINAMATH_CALUDE_centroid_trace_area_l134_13455


namespace NUMINAMATH_CALUDE_magnitude_of_AD_is_two_l134_13417

/-- Given two plane vectors m and n, prove that the magnitude of AD is 2 -/
theorem magnitude_of_AD_is_two (m n : ℝ × ℝ) : 
  let angle := Real.pi / 6
  let norm_m := Real.sqrt 3
  let norm_n := 2
  let AB := (2 * m.1 + 2 * n.1, 2 * m.2 + 2 * n.2)
  let AC := (2 * m.1 - 6 * n.1, 2 * m.2 - 6 * n.2)
  let D := ((AB.1 + AC.1) / 2, (AB.2 + AC.2) / 2)  -- midpoint of BC
  let AD := (D.1 - m.1, D.2 - m.2)
  Real.cos angle = Real.sqrt 3 / 2 →   -- angle between m and n
  norm_m = Real.sqrt 3 →
  norm_n = 2 →
  Real.sqrt (AD.1 ^ 2 + AD.2 ^ 2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_magnitude_of_AD_is_two_l134_13417


namespace NUMINAMATH_CALUDE_fraction_simplification_l134_13400

theorem fraction_simplification (a : ℝ) (h : a ≠ 1) : a / (a - 1) + 1 / (1 - a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l134_13400


namespace NUMINAMATH_CALUDE_integer_solution_equation_l134_13444

theorem integer_solution_equation (k x : ℤ) : 
  (Real.sqrt (39 - 6 * Real.sqrt 12) + Real.sqrt (k * x * (k * x + Real.sqrt 12) + 3) = 2 * k) → 
  (k = 3 ∨ k = 6) := by
sorry

end NUMINAMATH_CALUDE_integer_solution_equation_l134_13444


namespace NUMINAMATH_CALUDE_complex_product_ab_l134_13426

theorem complex_product_ab (a b : ℝ) (i : ℂ) (h : i * i = -1) 
  (h1 : (1 - 2*i)*i = a + b*i) : a * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_ab_l134_13426


namespace NUMINAMATH_CALUDE_number_divided_by_three_l134_13451

theorem number_divided_by_three : ∃ x : ℤ, x / 3 = x - 24 ∧ x = 72 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_three_l134_13451


namespace NUMINAMATH_CALUDE_b_squared_is_zero_matrix_l134_13456

theorem b_squared_is_zero_matrix (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : B ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_b_squared_is_zero_matrix_l134_13456


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l134_13439

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (Complex.I - 1)^2 + 4 / (Complex.I + 1)
  (z.im = -3) := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l134_13439


namespace NUMINAMATH_CALUDE_quadratic_function_increasing_on_positive_x_l134_13431

theorem quadratic_function_increasing_on_positive_x 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = x₁^2 - 1) 
  (h2 : y₂ = x₂^2 - 1) 
  (h3 : 0 < x₁) 
  (h4 : x₁ < x₂) : 
  y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_increasing_on_positive_x_l134_13431


namespace NUMINAMATH_CALUDE_inhabitable_earth_fraction_l134_13478

-- Define the fraction of Earth's surface that is land
def land_fraction : ℚ := 1 / 5

-- Define the fraction of land that is inhabitable
def inhabitable_land_fraction : ℚ := 1 / 3

-- Theorem: The fraction of Earth's surface that humans can live on is 1/15
theorem inhabitable_earth_fraction :
  land_fraction * inhabitable_land_fraction = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_inhabitable_earth_fraction_l134_13478


namespace NUMINAMATH_CALUDE_unique_solution_system_l134_13486

theorem unique_solution_system (s t : ℝ) : 
  15 * s + 10 * t = 270 ∧ s = 3 * t - 4 → s = 14 ∧ t = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l134_13486


namespace NUMINAMATH_CALUDE_exterior_angle_measure_l134_13443

theorem exterior_angle_measure (n : ℕ) (h : n > 2) :
  (n - 2) * 180 = 1260 →
  360 / n = 40 := by
sorry

end NUMINAMATH_CALUDE_exterior_angle_measure_l134_13443


namespace NUMINAMATH_CALUDE_arithmetic_sum_1000_l134_13461

theorem arithmetic_sum_1000 : 
  ∀ m n : ℕ+, 
    (Finset.sum (Finset.range (m + 1)) (λ i => n + i) = 1000) ↔ 
    ((m = 4 ∧ n = 198) ∨ (m = 24 ∧ n = 28)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_1000_l134_13461


namespace NUMINAMATH_CALUDE_number_manipulation_l134_13405

theorem number_manipulation (x : ℝ) : (x - 5) / 7 = 7 → (x - 6) / 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_manipulation_l134_13405


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l134_13423

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (h2 : a^2 + b^2 + c^2 = 16) 
  (h3 : a*b + b*c + c*a = 9) 
  (h4 : a^2 + b^2 = 10) : 
  a + b + c = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l134_13423


namespace NUMINAMATH_CALUDE_hundredthOddPositiveInteger_l134_13430

/-- The nth odd positive integer -/
def nthOddPositiveInteger (n : ℕ) : ℕ := 2 * n - 1

/-- Theorem: The 100th odd positive integer is 199 -/
theorem hundredthOddPositiveInteger : nthOddPositiveInteger 100 = 199 := by
  sorry

end NUMINAMATH_CALUDE_hundredthOddPositiveInteger_l134_13430


namespace NUMINAMATH_CALUDE_leyden_quadruple_theorem_l134_13412

/-- Definition of a Leyden quadruple -/
structure LeydenQuadruple where
  p : ℕ
  a₁ : ℕ
  a₂ : ℕ
  a₃ : ℕ

/-- The main theorem about Leyden quadruples -/
theorem leyden_quadruple_theorem (q : LeydenQuadruple) :
  (q.a₁ + q.a₂ + q.a₃) / 3 = q.p + 2 ↔ q.p = 5 := by
  sorry

end NUMINAMATH_CALUDE_leyden_quadruple_theorem_l134_13412


namespace NUMINAMATH_CALUDE_trig_identity_proof_l134_13436

theorem trig_identity_proof : 
  Real.sin (410 * π / 180) * Real.sin (550 * π / 180) - 
  Real.sin (680 * π / 180) * Real.cos (370 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l134_13436


namespace NUMINAMATH_CALUDE_basketball_game_score_l134_13463

theorem basketball_game_score (a b k d : ℕ) : 
  a = b →  -- Tied at the end of first quarter
  (4*a + 14*k = 4*b + 6*d + 2) →  -- Eagles won by two points
  (4*a + 14*k ≤ 100) →  -- Eagles scored no more than 100
  (4*b + 6*d ≤ 100) →  -- Panthers scored no more than 100
  (2*a + k) + (2*b + d) = 59 := by
sorry

end NUMINAMATH_CALUDE_basketball_game_score_l134_13463
