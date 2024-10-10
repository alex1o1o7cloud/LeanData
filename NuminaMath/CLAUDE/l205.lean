import Mathlib

namespace equation_solution_l205_20588

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    x₁ > 0 ∧ x₂ > 0 ∧
    (∀ (x : ℝ), x > 0 → 
      ((1/3) * (4*x^2 - 3) = (x^2 - 75*x - 15) * (x^2 + 40*x + 8)) ↔ 
      (x = x₁ ∨ x = x₂)) ∧
    x₁ = (75 + Real.sqrt 5677) / 2 ∧
    x₂ = (-40 + Real.sqrt 1572) / 2 :=
by sorry

end equation_solution_l205_20588


namespace quadratic_equation_m_value_l205_20561

/-- Given that (m-2)x^|m| - bx - 1 = 0 is a quadratic equation in x, prove that m = -2 -/
theorem quadratic_equation_m_value (m b : ℝ) : 
  (∀ x, ∃ a c : ℝ, (m - 2) * x^(|m|) - b*x - 1 = a*x^2 + b*x + c) → 
  m = -2 :=
by sorry

end quadratic_equation_m_value_l205_20561


namespace combination_count_l205_20591

theorem combination_count (n k m : ℕ) :
  (∃ (s : Finset (Finset ℕ)),
    (∀ t ∈ s, t.card = k ∧
      (∀ j ∈ t, 1 ≤ j ∧ j ≤ n) ∧
      (∀ (i j : ℕ), i ∈ t → j ∈ t → i < j → m ≤ j - i) ∧
      (∀ (i j : ℕ), i ∈ t → j ∈ t → i ≠ j → i < j)) ∧
    s.card = Nat.choose (n - (k - 1) * (m - 1)) k) :=
by sorry

end combination_count_l205_20591


namespace quadratic_properties_l205_20504

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x - 1)^2 + 2

-- Theorem stating the properties of the quadratic function
theorem quadratic_properties :
  (∃ (a : ℝ), ∀ (x : ℝ), f x = f (2 - x)) ∧ 
  (∀ (x : ℝ), f x ≤ 2) ∧
  (f 1 = 2) := by
  sorry

end quadratic_properties_l205_20504


namespace max_a1_value_l205_20518

/-- A sequence of non-negative real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n ≥ 0) ∧
  (∀ n ≥ 2, a (n + 1) = a n - a (n - 1) + n)

theorem max_a1_value (a : ℕ → ℝ) (h : RecurrenceSequence a) (h2022 : a 2 * a 2022 = 1) :
  ∃ (max_a1 : ℝ), a 1 ≤ max_a1 ∧ max_a1 = 4051 / 2025 :=
sorry

end max_a1_value_l205_20518


namespace distance_between_x_intercepts_l205_20523

-- Define the slopes and intersection point
def m1 : ℝ := 4
def m2 : ℝ := -2
def intersection : ℝ × ℝ := (8, 20)

-- Define the lines using point-slope form
def line1 (x : ℝ) : ℝ := m1 * (x - intersection.1) + intersection.2
def line2 (x : ℝ) : ℝ := m2 * (x - intersection.1) + intersection.2

-- Define x-intercepts
noncomputable def x_intercept1 : ℝ := (intersection.2 - m1 * intersection.1) / (-m1)
noncomputable def x_intercept2 : ℝ := (intersection.2 - m2 * intersection.1) / (-m2)

-- Theorem statement
theorem distance_between_x_intercepts :
  |x_intercept2 - x_intercept1| = 15 := by sorry

end distance_between_x_intercepts_l205_20523


namespace equation_solutions_l205_20582

/-- The integer part of a real number -/
noncomputable def intPart (x : ℝ) : ℤ := Int.floor x

/-- The fractional part of a real number -/
noncomputable def fracPart (x : ℝ) : ℝ := x - Int.floor x

/-- The main theorem stating the solutions to the equation -/
theorem equation_solutions :
  ∀ x : ℝ, 
  (intPart x : ℝ) * fracPart x + x = 2 * fracPart x + 9 →
  (x = 9 ∨ x = 8 + 1/7 ∨ x = 7 + 1/3) :=
by sorry

end equation_solutions_l205_20582


namespace chess_go_problem_l205_20550

/-- Represents the prices and quantities of Chinese chess and Go sets -/
structure ChessGoSets where
  chess_price : ℝ
  go_price : ℝ
  total_sets : ℕ
  max_cost : ℝ

/-- Defines the conditions given in the problem -/
def problem_conditions (s : ChessGoSets) : Prop :=
  2 * s.chess_price + 3 * s.go_price = 140 ∧
  4 * s.chess_price + s.go_price = 130 ∧
  s.total_sets = 80 ∧
  s.max_cost = 2250

/-- Theorem stating the solution to the problem -/
theorem chess_go_problem (s : ChessGoSets) 
  (h : problem_conditions s) : 
  s.chess_price = 25 ∧ 
  s.go_price = 30 ∧ 
  (∀ m : ℕ, m * s.go_price + (s.total_sets - m) * s.chess_price ≤ s.max_cost → m ≤ 50) ∧
  (∀ a : ℝ, a > 0 → 
    (a < 10 → 0.9 * a * s.go_price < 0.7 * a * s.go_price + 60) ∧
    (a = 10 → 0.9 * a * s.go_price = 0.7 * a * s.go_price + 60) ∧
    (a > 10 → 0.9 * a * s.go_price > 0.7 * a * s.go_price + 60)) :=
by
  sorry

end chess_go_problem_l205_20550


namespace parabola_hyperbola_intersection_l205_20586

/-- Given a parabola and a hyperbola with specific properties, prove that the parameter 'a' of the hyperbola equals 1/4. -/
theorem parabola_hyperbola_intersection (p : ℝ) (m : ℝ) (a : ℝ) : 
  p > 0 → -- p is positive
  m^2 = 2*p -- point (1,m) is on the parabola y^2 = 2px
  → (1 - p/2)^2 + m^2 = 5^2 -- distance from (1,m) to focus (p/2, 0) is 5
  → ∃ (k : ℝ), k^2 * a = 1 ∧ k * m = 2 -- asymptote y = kx is perpendicular to AM (slope of AM is m/2)
  → a = 1/4 := by sorry

end parabola_hyperbola_intersection_l205_20586


namespace export_probabilities_l205_20565

/-- The number of inspections required for each batch -/
def num_inspections : ℕ := 5

/-- The probability of failing any given inspection -/
def fail_prob : ℝ := 0.2

/-- The probability of passing any given inspection -/
def pass_prob : ℝ := 1 - fail_prob

/-- The probability that a batch cannot be exported -/
def cannot_export_prob : ℝ := 1 - (pass_prob ^ num_inspections + num_inspections * fail_prob * pass_prob ^ (num_inspections - 1))

/-- The probability that all five inspections must be completed -/
def all_inspections_prob : ℝ := (num_inspections - 1) * fail_prob * pass_prob ^ (num_inspections - 2)

theorem export_probabilities :
  (cannot_export_prob = 0.26) ∧ (all_inspections_prob = 0.41) := by
  sorry

end export_probabilities_l205_20565


namespace shepherd_problem_l205_20553

def checkpoint (n : ℕ) : ℕ := n / 2 + 1

def process (initial : ℕ) (checkpoints : ℕ) : ℕ :=
  match checkpoints with
  | 0 => initial
  | n + 1 => checkpoint (process initial n)

theorem shepherd_problem (initial : ℕ) (checkpoints : ℕ) :
  initial = 254 ∧ checkpoints = 6 → process initial checkpoints = 2 := by
  sorry

end shepherd_problem_l205_20553


namespace inscribed_circle_radius_squared_l205_20540

/-- A circle inscribed in a quadrilateral EFGH -/
structure InscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The circle is tangent to EF at R -/
  ER : ℝ
  /-- The circle is tangent to EF at R -/
  RF : ℝ
  /-- The circle is tangent to GH at S -/
  GS : ℝ
  /-- The circle is tangent to GH at S -/
  SH : ℝ

/-- The theorem stating that the square of the radius of the inscribed circle is 1357 -/
theorem inscribed_circle_radius_squared (c : InscribedCircle)
  (h1 : c.ER = 15)
  (h2 : c.RF = 31)
  (h3 : c.GS = 47)
  (h4 : c.SH = 29) :
  c.r ^ 2 = 1357 := by
  sorry

end inscribed_circle_radius_squared_l205_20540


namespace star_value_l205_20525

/-- Operation star defined as a * b = 3a - b^3 -/
def star (a b : ℝ) : ℝ := 3 * a - b^3

/-- Theorem: If a * 3 = 63, then a = 30 -/
theorem star_value (a : ℝ) (h : star a 3 = 63) : a = 30 := by
  sorry

end star_value_l205_20525


namespace smallest_zero_difference_l205_20552

def u (n : ℕ) : ℤ := n^3 - n

def finite_difference (f : ℕ → ℤ) (k : ℕ) : ℕ → ℤ :=
  match k with
  | 0 => f
  | k+1 => λ n => finite_difference f k (n+1) - finite_difference f k n

theorem smallest_zero_difference :
  (∃ k : ℕ, ∀ n : ℕ, finite_difference u k n = 0) ∧
  (∀ k : ℕ, k < 4 → ∃ n : ℕ, finite_difference u k n ≠ 0) ∧
  (∀ n : ℕ, finite_difference u 4 n = 0) := by
  sorry

end smallest_zero_difference_l205_20552


namespace y2k_game_second_player_strategy_l205_20508

/-- Represents a player in the Y2K Game -/
inductive Player : Type
  | First : Player
  | Second : Player

/-- Represents a letter that can be placed on the board -/
inductive Letter : Type
  | S : Letter
  | O : Letter

/-- Represents the state of a square on the board -/
inductive Square : Type
  | Empty : Square
  | Filled : Letter → Square

/-- Represents the game board -/
def Board : Type := Fin 2000 → Square

/-- Represents a move in the game -/
structure Move where
  position : Fin 2000
  letter : Letter

/-- Represents the game state -/
structure GameState where
  board : Board
  currentPlayer : Player

/-- Represents a strategy for a player -/
def Strategy : Type := GameState → Move

/-- Checks if a player has won the game -/
def hasWon (board : Board) (player : Player) : Prop := sorry

/-- Checks if the game is a draw -/
def isDraw (board : Board) : Prop := sorry

/-- The Y2K Game theorem -/
theorem y2k_game_second_player_strategy :
  ∃ (strategy : Strategy),
    ∀ (initialState : GameState),
      initialState.currentPlayer = Player.Second →
        (∃ (finalState : GameState),
          (hasWon finalState.board Player.Second ∨ isDraw finalState.board)) :=
sorry

end y2k_game_second_player_strategy_l205_20508


namespace coefficient_equals_nth_term_l205_20516

def a (n : ℕ) : ℕ := 3 * n - 5

theorem coefficient_equals_nth_term :
  let coefficient : ℕ := (Nat.choose 5 4) + (Nat.choose 6 4) + (Nat.choose 7 4)
  coefficient = a 20 := by sorry

end coefficient_equals_nth_term_l205_20516


namespace harrys_seed_purchase_l205_20536

/-- Represents the number of packets of each seed type and the total spent -/
structure SeedPurchase where
  pumpkin : ℕ
  tomato : ℕ
  chili : ℕ
  total_spent : ℚ

/-- Calculates the total cost of a seed purchase -/
def calculate_total_cost (purchase : SeedPurchase) : ℚ :=
  2.5 * purchase.pumpkin + 1.5 * purchase.tomato + 0.9 * purchase.chili

/-- Theorem stating that Harry's purchase of 3 pumpkin, 4 tomato, and 5 chili pepper seed packets
    totaling $18 is correct -/
theorem harrys_seed_purchase :
  ∃ (purchase : SeedPurchase),
    purchase.pumpkin = 3 ∧
    purchase.tomato = 4 ∧
    purchase.chili = 5 ∧
    purchase.total_spent = 18 ∧
    calculate_total_cost purchase = purchase.total_spent :=
  sorry

end harrys_seed_purchase_l205_20536


namespace correct_yeast_counting_operation_l205_20539

/-- Represents an experimental operation -/
inductive ExperimentalOperation
  | YeastCounting
  | PigmentSeparation
  | AuxinRooting
  | Plasmolysis

/-- Determines if an experimental operation is correct -/
def is_correct_operation (op : ExperimentalOperation) : Prop :=
  match op with
  | ExperimentalOperation.YeastCounting => true
  | _ => false

/-- Theorem stating that shaking the culture solution before yeast counting is the correct operation -/
theorem correct_yeast_counting_operation :
  is_correct_operation ExperimentalOperation.YeastCounting := by
  sorry

end correct_yeast_counting_operation_l205_20539


namespace folding_problem_l205_20513

/-- Represents the folding rate for each type of clothing --/
structure FoldingRate where
  shirts : ℕ
  pants : ℕ
  shorts : ℕ

/-- Represents the number of items for each type of clothing --/
structure ClothingItems where
  shirts : ℕ
  pants : ℕ
  shorts : ℕ

/-- Calculates the remaining items to be folded given the initial conditions --/
def remainingItems (initialItems : ClothingItems) (rate : FoldingRate) (totalTime : ℕ) 
    (shirtFoldTime : ℕ) (pantFoldTime : ℕ) (shirtBreakTime : ℕ) (pantBreakTime : ℕ) : ClothingItems :=
  sorry

/-- The main theorem to be proved --/
theorem folding_problem (initialItems : ClothingItems) (rate : FoldingRate) (totalTime : ℕ) 
    (shirtFoldTime : ℕ) (pantFoldTime : ℕ) (shirtBreakTime : ℕ) (pantBreakTime : ℕ) :
    initialItems = ClothingItems.mk 30 15 20 ∧ 
    rate = FoldingRate.mk 12 8 10 ∧
    totalTime = 120 ∧
    shirtFoldTime = 45 ∧
    pantFoldTime = 30 ∧
    shirtBreakTime = 15 ∧
    pantBreakTime = 10 →
    remainingItems initialItems rate totalTime shirtFoldTime pantFoldTime shirtBreakTime pantBreakTime = 
    ClothingItems.mk 21 11 17 :=
  sorry

end folding_problem_l205_20513


namespace boys_neither_happy_nor_sad_boys_neither_happy_nor_sad_is_6_l205_20556

theorem boys_neither_happy_nor_sad (total_children : Nat) (happy_children : Nat) (sad_children : Nat) 
  (neither_children : Nat) (total_boys : Nat) (total_girls : Nat) (happy_boys : Nat) (sad_girls : Nat) : Nat :=
  by
  -- Assumptions
  have h1 : total_children = 60 := by sorry
  have h2 : happy_children = 30 := by sorry
  have h3 : sad_children = 10 := by sorry
  have h4 : neither_children = 20 := by sorry
  have h5 : total_boys = 18 := by sorry
  have h6 : total_girls = 42 := by sorry
  have h7 : happy_boys = 6 := by sorry
  have h8 : sad_girls = 4 := by sorry
  
  -- Proof
  sorry

-- The theorem statement
theorem boys_neither_happy_nor_sad_is_6 : 
  boys_neither_happy_nor_sad 60 30 10 20 18 42 6 4 = 6 := by sorry

end boys_neither_happy_nor_sad_boys_neither_happy_nor_sad_is_6_l205_20556


namespace five_by_five_grid_properties_l205_20524

/-- Represents a square grid --/
structure Grid :=
  (size : ℕ)

/-- Counts the number of squares in a grid --/
def count_squares (g : Grid) : ℕ :=
  sorry

/-- Counts the number of pairs of parallel lines in a grid --/
def count_parallel_pairs (g : Grid) : ℕ :=
  sorry

/-- Counts the number of rectangles in a grid --/
def count_rectangles (g : Grid) : ℕ :=
  sorry

/-- Theorem stating the properties of a 5x5 grid --/
theorem five_by_five_grid_properties :
  let g : Grid := ⟨5⟩
  count_squares g = 55 ∧
  count_parallel_pairs g = 30 ∧
  count_rectangles g = 225 :=
by sorry

end five_by_five_grid_properties_l205_20524


namespace blueberry_count_l205_20562

/-- Represents the number of berries in a box of a specific color -/
structure BerryBox where
  blue : ℕ
  red : ℕ
  green : ℕ

/-- The change in berry counts when replacing boxes -/
structure BerryChange where
  total : ℤ
  difference : ℤ

theorem blueberry_count (box : BerryBox) 
  (replace_blue_with_red : BerryChange)
  (replace_green_with_blue : BerryChange) :
  (replace_blue_with_red.total = 10) →
  (replace_blue_with_red.difference = 50) →
  (replace_green_with_blue.total = -5) →
  (replace_green_with_blue.difference = -30) →
  (box.red - box.blue = replace_blue_with_red.total) →
  (box.blue - box.green = -replace_green_with_blue.total) →
  (box.green - 2 * box.blue = -replace_green_with_blue.difference) →
  box.blue = 35 := by
  sorry

end blueberry_count_l205_20562


namespace ratio_michael_monica_l205_20527

-- Define the ages as real numbers
variable (patrick_age michael_age monica_age : ℝ)

-- Define the conditions
axiom ratio_patrick_michael : patrick_age / michael_age = 3 / 5
axiom sum_of_ages : patrick_age + michael_age + monica_age = 245
axiom age_difference : monica_age - patrick_age = 80

-- Theorem to prove
theorem ratio_michael_monica :
  michael_age / monica_age = 3 / 5 :=
sorry

end ratio_michael_monica_l205_20527


namespace closed_path_theorem_l205_20531

/-- A closed path on an m×n table satisfying specific conditions -/
structure ClosedPath (m n : ℕ) where
  -- Ensure m and n are at least 4
  m_ge_four : m ≥ 4
  n_ge_four : n ≥ 4
  -- A is the number of straight-forward vertices
  A : ℕ
  -- B is the number of squares with two opposite sides used
  B : ℕ
  -- C is the number of unused squares
  C : ℕ
  -- The path doesn't intersect itself
  no_self_intersection : True
  -- The path passes through all interior vertices
  passes_all_interior : True
  -- The path doesn't pass through outer vertices
  no_outer_vertices : True

/-- Theorem: For a closed path on an m×n table satisfying the given conditions,
    A = B - C + m + n - 1 -/
theorem closed_path_theorem (m n : ℕ) (path : ClosedPath m n) :
  path.A = path.B - path.C + m + n - 1 := by
  sorry

end closed_path_theorem_l205_20531


namespace arithmetic_mean_difference_l205_20571

theorem arithmetic_mean_difference (p q r : ℝ) 
  (mean_pq : (p + q) / 2 = 10)
  (mean_qr : (q + r) / 2 = 22) :
  r - p = 24 := by
sorry

end arithmetic_mean_difference_l205_20571


namespace group_size_proof_l205_20568

theorem group_size_proof (total_paise : ℕ) (h : total_paise = 4624) :
  ∃ n : ℕ, n * n = total_paise ∧ n = 68 := by
  sorry

end group_size_proof_l205_20568


namespace special_numbers_property_l205_20549

/-- Given a natural number, return the sum of its digits -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The list of 13 numbers that satisfy the conditions -/
def specialNumbers : List ℕ := [6, 15, 24, 33, 42, 51, 60, 105, 114, 123, 132, 141, 150]

theorem special_numbers_property :
  (∃ (nums : List ℕ),
    nums.length = 13 ∧
    nums.sum = 996 ∧
    nums.Nodup ∧
    ∀ (x y : ℕ), x ∈ nums → y ∈ nums → digitSum x = digitSum y) := by
  sorry

end special_numbers_property_l205_20549


namespace complex_equation_solution_l205_20534

theorem complex_equation_solution :
  ∃ (z : ℂ), 2 - (3 + Complex.I) * z = 1 - (3 - Complex.I) * z ∧ z = Complex.I / 2 := by
  sorry

end complex_equation_solution_l205_20534


namespace point_in_fourth_quadrant_l205_20563

/-- A point in the Cartesian plane is in the fourth quadrant if its x-coordinate is positive and its y-coordinate is negative -/
def fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

/-- The point (2, -4) is in the fourth quadrant -/
theorem point_in_fourth_quadrant : fourth_quadrant (2, -4) := by
  sorry

end point_in_fourth_quadrant_l205_20563


namespace function_defined_for_all_reals_l205_20595

/-- The function f(t) is defined for all real numbers t. -/
theorem function_defined_for_all_reals :
  ∀ t : ℝ, ∃ y : ℝ, y = 1 / ((t - 1)^2 + (t + 1)^2) := by
  sorry

end function_defined_for_all_reals_l205_20595


namespace square_of_difference_l205_20535

theorem square_of_difference (y : ℝ) (h : y^2 ≥ 49) :
  (7 - Real.sqrt (y^2 - 49))^2 = y^2 - 14 * Real.sqrt (y^2 - 49) := by
  sorry

end square_of_difference_l205_20535


namespace parabola_intersection_l205_20501

/-- Parabola 1 function -/
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 1

/-- Parabola 2 function -/
def g (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 1

/-- Theorem stating that (0, 1) and (-8, 233) are the only intersection points -/
theorem parabola_intersection :
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = 0 ∧ y = 1) ∨ (x = -8 ∧ y = 233) := by
  sorry

#check parabola_intersection

end parabola_intersection_l205_20501


namespace milk_production_l205_20573

theorem milk_production (y : ℝ) (h : y > 0) : 
  let initial_production := (y + 2) / (y * (y + 3))
  let new_cows := y + 4
  let new_milk := y + 6
  (new_milk / (new_cows * initial_production)) = (y * (y + 3) * (y + 6)) / ((y + 2) * (y + 4)) :=
by sorry

end milk_production_l205_20573


namespace quadratic_inequality_range_l205_20528

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) := by sorry

end quadratic_inequality_range_l205_20528


namespace repeating_decimal_division_l205_20547

theorem repeating_decimal_division (a b : ℚ) :
  a = 45 / 99 →
  b = 18 / 99 →
  a / b = 5 / 2 := by
  sorry

end repeating_decimal_division_l205_20547


namespace no_integer_distances_point_l205_20585

theorem no_integer_distances_point (a b : ℕ) (ha : Odd a) (hb : Odd b) :
  ¬ ∃ (x y : ℚ), 0 < x ∧ x < b ∧ 0 < y ∧ y < a ∧
    (∀ (i j : ℕ), i ≤ 1 ∧ j ≤ 1 →
      ∃ (n : ℕ), (x - i * b)^2 + (y - j * a)^2 = n^2) :=
by sorry

end no_integer_distances_point_l205_20585


namespace strawberry_sugar_purchase_strategy_l205_20509

theorem strawberry_sugar_purchase_strategy :
  -- Define constants
  let discount_threshold : ℝ := 1000
  let discount_rate : ℝ := 0.5
  let budget : ℝ := 1200
  let strawberry_price : ℝ := 300
  let sugar_price : ℝ := 30
  let strawberry_amount : ℝ := 4
  let sugar_amount : ℝ := 6

  -- Define purchase strategy
  let first_purchase_strawberry : ℝ := 3
  let first_purchase_sugar : ℝ := 4
  let second_purchase_strawberry : ℝ := strawberry_amount - first_purchase_strawberry
  let second_purchase_sugar : ℝ := sugar_amount - first_purchase_sugar

  -- Calculate costs
  let first_purchase_cost : ℝ := first_purchase_strawberry * strawberry_price + first_purchase_sugar * sugar_price
  let second_purchase_full_price : ℝ := second_purchase_strawberry * strawberry_price + second_purchase_sugar * sugar_price
  let second_purchase_discounted : ℝ := second_purchase_full_price * (1 - discount_rate)
  let total_cost : ℝ := first_purchase_cost + second_purchase_discounted

  -- Theorem statement
  (first_purchase_cost ≥ discount_threshold) →
  (total_cost ≤ budget) ∧
  (first_purchase_strawberry + second_purchase_strawberry = strawberry_amount) ∧
  (first_purchase_sugar + second_purchase_sugar = sugar_amount) :=
by sorry

end strawberry_sugar_purchase_strategy_l205_20509


namespace find_first_number_l205_20502

theorem find_first_number (x : ℝ) : 
  let set1 := [20, 40, 60]
  let set2 := [x, 70, 19]
  (set1.sum / set1.length) = (set2.sum / set2.length) + 7 →
  x = 10 := by
sorry

end find_first_number_l205_20502


namespace scavenger_hunt_ratio_l205_20580

theorem scavenger_hunt_ratio : 
  ∀ (lewis samantha tanya : ℕ),
  lewis = samantha + 4 →
  ∃ k : ℕ, samantha = k * tanya →
  tanya = 4 →
  lewis = 20 →
  samantha / tanya = 4 := by
sorry

end scavenger_hunt_ratio_l205_20580


namespace ellipse_foci_distance_l205_20537

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of the ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ :=
  sorry

theorem ellipse_foci_distance :
  let e : ParallelAxisEllipse := ⟨(6, 0), (0, 2)⟩
  foci_distance e = 4 * Real.sqrt 2 := by
  sorry

end ellipse_foci_distance_l205_20537


namespace train_length_calculation_l205_20503

/-- Calculates the length of a train given its speed and time to cross a point. -/
def trainLength (speed : Real) (time : Real) : Real :=
  speed * time

theorem train_length_calculation (speed : Real) (time : Real) 
  (h1 : speed = 90 * 1000 / 3600) -- Speed in m/s
  (h2 : time = 20) : -- Time in seconds
  trainLength speed time = 500 := by
  sorry

#check train_length_calculation

end train_length_calculation_l205_20503


namespace complex_number_imaginary_part_l205_20596

theorem complex_number_imaginary_part (z : ℂ) (h : (1 + z) / (1 - z) = I) : z.im = 1 := by
  sorry

end complex_number_imaginary_part_l205_20596


namespace calculate_markup_l205_20520

/-- Calculates the markup for an article given its purchase price, overhead percentage, and desired net profit. -/
theorem calculate_markup (purchase_price overhead_percent net_profit : ℚ) : 
  purchase_price = 48 →
  overhead_percent = 15 / 100 →
  net_profit = 12 →
  purchase_price + overhead_percent * purchase_price + net_profit - purchase_price = 19.2 := by
  sorry

end calculate_markup_l205_20520


namespace semicircle_pattern_area_l205_20579

/-- The area of the shaded region formed by semicircles in a foot-long pattern -/
theorem semicircle_pattern_area :
  let diameter : ℝ := 3  -- diameter of each semicircle in inches
  let pattern_length : ℝ := 12  -- length of the pattern in inches (1 foot)
  let num_semicircles : ℝ := pattern_length / diameter  -- number of semicircles in the pattern
  let semicircle_area : ℝ → ℝ := λ r => (π * r^2) / 2  -- area of a semicircle
  let total_area : ℝ := num_semicircles * semicircle_area (diameter / 2)
  total_area = (9/2) * π
  := by sorry

end semicircle_pattern_area_l205_20579


namespace right_triangle_third_side_l205_20542

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = 6 ∧ b = 8) ∨ (a = 6 ∧ c = 8) ∨ (b = 6 ∧ c = 8) →
  (a^2 + b^2 = c^2) →
  c = 10 ∨ c = 2 * Real.sqrt 7 :=
by sorry

end right_triangle_third_side_l205_20542


namespace uncle_ben_eggs_l205_20511

theorem uncle_ben_eggs (total_chickens roosters non_laying_hens eggs_per_hen : ℕ) 
  (h1 : total_chickens = 440)
  (h2 : roosters = 39)
  (h3 : non_laying_hens = 15)
  (h4 : eggs_per_hen = 3) :
  total_chickens - roosters - non_laying_hens = 386 →
  (total_chickens - roosters - non_laying_hens) * eggs_per_hen = 1158 := by
  sorry

end uncle_ben_eggs_l205_20511


namespace black_and_white_films_count_l205_20589

theorem black_and_white_films_count 
  (B : ℕ) -- number of black-and-white films
  (x y : ℚ) -- parameters for selection percentage and color films
  (h1 : y / x > 0) -- ensure y/x is positive
  (h2 : y > 0) -- ensure y is positive
  (h3 : (4 * y) / ((y / x * B / 100) + 4 * y) = 10 / 11) -- fraction of selected color films
  : B = 40 * x := by
sorry

end black_and_white_films_count_l205_20589


namespace horner_method_f_neg_two_l205_20530

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem horner_method_f_neg_two :
  f (-2) = -1 := by sorry

end horner_method_f_neg_two_l205_20530


namespace total_amount_is_120_l205_20594

def amount_from_grandpa : ℕ := 30

def amount_from_grandma : ℕ := 3 * amount_from_grandpa

def total_amount : ℕ := amount_from_grandpa + amount_from_grandma

theorem total_amount_is_120 : total_amount = 120 := by
  sorry

end total_amount_is_120_l205_20594


namespace triangle_abc_proof_l205_20581

theorem triangle_abc_proof (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * Real.cos A * Real.cos B - b * Real.sin A * Real.sin A - c * Real.cos A = 2 * b * Real.cos B →
  b = Real.sqrt 7 * a →
  (1/2) * a * c * Real.sin B = 2 * Real.sqrt 3 →
  B = 2 * π / 3 ∧ a = 2 := by
  sorry

end triangle_abc_proof_l205_20581


namespace four_digit_numbers_count_l205_20558

theorem four_digit_numbers_count : 
  (Finset.range 9000).card = (Finset.Icc 1000 9999).card := by sorry

end four_digit_numbers_count_l205_20558


namespace pure_imaginary_condition_l205_20572

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (a + 1) * (a - 1 + Complex.I) = Complex.I * b) → a = 1 := by
  sorry

end pure_imaginary_condition_l205_20572


namespace gcd_lcm_product_24_60_l205_20541

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end gcd_lcm_product_24_60_l205_20541


namespace expression_simplification_and_evaluation_l205_20500

theorem expression_simplification_and_evaluation (x : ℤ) 
  (h1 : 3 * x + 7 > 1) (h2 : 2 * x - 1 < 5) :
  let expr := (x / (x - 1)) / ((x^2 - x) / (x^2 - 2*x + 1)) - (x + 2) / (x + 1)
  (expr = -1 / (x + 1)) ∧ 
  (expr = -1/3 ∨ expr = -1/2 ∨ expr = -1) := by
  sorry

end expression_simplification_and_evaluation_l205_20500


namespace fraction_simplification_l205_20519

theorem fraction_simplification : (5 * 8) / 10 = 4 := by
  sorry

end fraction_simplification_l205_20519


namespace five_digit_divisibility_l205_20583

/-- A five-digit number -/
def FiveDigitNumber : Type := { n : ℕ // 10000 ≤ n ∧ n < 100000 }

/-- A four-digit number -/
def FourDigitNumber : Type := { n : ℕ // 1000 ≤ n ∧ n < 10000 }

/-- Extract the four-digit number from a five-digit number by removing the middle digit -/
def extractFourDigit (n : FiveDigitNumber) : FourDigitNumber :=
  sorry

theorem five_digit_divisibility (n : FiveDigitNumber) :
  (∃ (m : FourDigitNumber), m = extractFourDigit n ∧ n.val % m.val = 0) ↔ n.val % 1000 = 0 :=
sorry

end five_digit_divisibility_l205_20583


namespace equation_solution_l205_20507

theorem equation_solution (x y : ℚ) : 
  (4 * x + 2 * y = 12) → 
  (2 * x + 4 * y = 16) → 
  (20 * x^2 + 24 * x * y + 20 * y^2 = 3280 / 9) := by
sorry

end equation_solution_l205_20507


namespace fraction_comparison_l205_20545

theorem fraction_comparison : (22222222221 : ℚ) / 22222222223 > (33333333331 : ℚ) / 33333333334 := by
  sorry

end fraction_comparison_l205_20545


namespace squared_sum_bound_l205_20590

theorem squared_sum_bound (a b : ℝ) (x₁ x₂ : ℝ) : 
  (3 * x₁^2 + 3*(a+b)*x₁ + 4*a*b = 0) →
  (3 * x₂^2 + 3*(a+b)*x₂ + 4*a*b = 0) →
  (x₁ * (x₁ + 1) + x₂ * (x₂ + 1) = (x₁ + 1) * (x₂ + 1)) →
  (a + b)^2 ≤ 4 := by
sorry

end squared_sum_bound_l205_20590


namespace hay_in_final_mixture_l205_20564

/-- Represents the composition of a feed mixture -/
structure FeedMixture where
  oats : ℝ  -- Percentage of oats
  corn : ℝ  -- Percentage of corn
  hay : ℝ   -- Percentage of hay
  mass : ℝ  -- Mass of the mixture in kg

/-- Theorem stating the amount of hay in the final mixture -/
theorem hay_in_final_mixture
  (stepan : FeedMixture)
  (pavel : FeedMixture)
  (final : FeedMixture)
  (h1 : stepan.hay = 40)
  (h2 : pavel.oats = 26)
  (h3 : stepan.corn = pavel.corn)
  (h4 : stepan.mass = 150)
  (h5 : pavel.mass = 250)
  (h6 : final.corn = 30)
  (h7 : final.mass = stepan.mass + pavel.mass)
  (h8 : final.corn * final.mass = stepan.corn * stepan.mass + pavel.corn * pavel.mass) :
  final.hay * final.mass = 170 := by
  sorry

end hay_in_final_mixture_l205_20564


namespace square_differences_l205_20567

theorem square_differences (n : ℕ) : 
  (n + 1)^2 = n^2 + (2*n + 1) ∧ (n - 1)^2 = n^2 - (2*n - 1) :=
by sorry

end square_differences_l205_20567


namespace quadratic_function_with_equal_roots_l205_20533

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_with_equal_roots 
  (f : ℝ → ℝ) 
  (h1 : QuadraticFunction f)
  (h2 : ∀ x, deriv f x = 2 * x + 2)
  (h3 : ∃! r : ℝ, f r = 0 ∧ (deriv f r = 0)) :
  ∀ x, f x = x^2 + 2*x + 1 := by
  sorry

end quadratic_function_with_equal_roots_l205_20533


namespace geometric_sequence_ratio_l205_20532

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n, a n > 0) →
  geometric_sequence a q →
  (a 2 - a 1 = a 1 - (1/2) * a 3) →
  (a 3 + a 4) / (a 4 + a 5) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end geometric_sequence_ratio_l205_20532


namespace inequality_proof_l205_20575

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x / Real.sqrt y + y / Real.sqrt x ≥ Real.sqrt x + Real.sqrt y := by
  sorry

end inequality_proof_l205_20575


namespace chair_circumference_l205_20551

def parallelogram_circumference (side1 : ℝ) (side2 : ℝ) : ℝ :=
  2 * (side1 + side2)

theorem chair_circumference :
  let side1 := 18
  let side2 := 12
  parallelogram_circumference side1 side2 = 60 := by
sorry

end chair_circumference_l205_20551


namespace decimal_point_shift_l205_20510

theorem decimal_point_shift (x : ℝ) : x - x / 10 = 37.35 → x = 41.5 := by
  sorry

end decimal_point_shift_l205_20510


namespace largest_prime_diff_144_l205_20546

/-- Two natural numbers are considered different if they are not equal -/
def Different (a b : ℕ) : Prop := a ≠ b

/-- A natural number is prime if it's greater than 1 and its only positive divisors are 1 and itself -/
def IsPrime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d > 0 → d ∣ p → d = 1 ∨ d = p

/-- The statement that the largest possible difference between two different primes summing to 144 is 134 -/
theorem largest_prime_diff_144 : 
  ∃ (p q : ℕ), Different p q ∧ IsPrime p ∧ IsPrime q ∧ p + q = 144 ∧ 
  (∀ (r s : ℕ), Different r s → IsPrime r → IsPrime s → r + s = 144 → s - r ≤ 134) ∧
  q - p = 134 := by
sorry

end largest_prime_diff_144_l205_20546


namespace pentagon_area_greater_than_third_square_l205_20587

theorem pentagon_area_greater_than_third_square (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + (a*b)/4 + (Real.sqrt 3/4)*b^2 > ((a+b)^2)/3 := by
  sorry

end pentagon_area_greater_than_third_square_l205_20587


namespace area_of_region_l205_20517

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 20 ∧ 
   A = Real.pi * (Real.sqrt ((x + 5)^2 + (y - 2)^2))^2 ∧
   x^2 + y^2 + 10*x - 4*y + 9 = 0) := by
  sorry

end area_of_region_l205_20517


namespace intersection_A_complement_B_l205_20555

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | x^2 - 5*x + 4 < 0}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {0, 1} := by
  sorry

end intersection_A_complement_B_l205_20555


namespace min_mutually_visible_pairs_l205_20569

/-- A configuration of birds on a circle. -/
structure BirdConfiguration where
  /-- The total number of birds. -/
  total_birds : ℕ
  /-- The number of points on the circle where birds can sit. -/
  num_points : ℕ
  /-- The distribution of birds across the points. -/
  distribution : Fin num_points → ℕ
  /-- The sum of birds across all points equals the total number of birds. -/
  sum_constraint : (Finset.univ.sum distribution) = total_birds

/-- The number of mutually visible pairs in a given configuration. -/
def mutually_visible_pairs (config : BirdConfiguration) : ℕ :=
  Finset.sum Finset.univ (fun i => config.distribution i * (config.distribution i - 1) / 2)

/-- The theorem stating the minimum number of mutually visible pairs. -/
theorem min_mutually_visible_pairs :
  ∀ (config : BirdConfiguration),
    config.total_birds = 155 →
    mutually_visible_pairs config ≥ 270 :=
  sorry

end min_mutually_visible_pairs_l205_20569


namespace quadratic_max_condition_l205_20577

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 6*x - 7

-- Define the maximum value function
def y_max (t : ℝ) : ℝ := -(t-3)^2 + 2

-- Theorem statement
theorem quadratic_max_condition (t : ℝ) :
  (∀ x : ℝ, t ≤ x ∧ x ≤ t + 2 → f x ≤ y_max t) →
  (∃ x : ℝ, t ≤ x ∧ x ≤ t + 2 ∧ f x = y_max t) →
  t ≥ 3 :=
by sorry

end quadratic_max_condition_l205_20577


namespace quadratic_inequality_l205_20592

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) 
  (h1 : ∀ x, a * x^2 + b * x + c > 0 ↔ x < -2 ∨ x > 4) :
  f a b c 2 < f a b c (-1) ∧ f a b c (-1) < f a b c 5 := by
  sorry

end quadratic_inequality_l205_20592


namespace unique_stamp_value_l205_20526

/-- Given stamps of denominations 6, n, and n+1 cents, 
    this function checks if 115 cents is the greatest 
    postage that cannot be formed -/
def is_valid_stamp_set (n : ℕ) : Prop :=
  n > 0 ∧ 
  (∀ m : ℕ, m > 115 → ∃ a b c : ℕ, m = 6*a + n*b + (n+1)*c) ∧
  ¬(∃ a b c : ℕ, 115 = 6*a + n*b + (n+1)*c)

/-- The theorem stating that 24 is the only value of n 
    that satisfies the stamp condition -/
theorem unique_stamp_value : 
  (∃! n : ℕ, is_valid_stamp_set n) ∧ 
  (∀ n : ℕ, is_valid_stamp_set n → n = 24) :=
sorry

end unique_stamp_value_l205_20526


namespace right_triangle_area_l205_20578

/-- Given a right-angled triangle with perimeter 18 and sum of squares of side lengths 128, its area is 9. -/
theorem right_triangle_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 18 →
  a^2 + b^2 + c^2 = 128 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 9 :=
by sorry

end right_triangle_area_l205_20578


namespace log_inequality_solution_set_l205_20538

theorem log_inequality_solution_set :
  let f : ℝ → ℝ := fun x => Real.log (2 * x + 1) / Real.log (1/2)
  let S : Set ℝ := {x | f x ≥ Real.log 3 / Real.log (1/2)}
  S = Set.Ioc (-1/2) 1 :=
by sorry

end log_inequality_solution_set_l205_20538


namespace odd_function_value_l205_20514

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem odd_function_value (f : ℝ → ℝ) (h1 : is_odd f) (h2 : ∀ x < 0, f x = x^2 + 3*x) :
  f 2 = 2 := by sorry

end odd_function_value_l205_20514


namespace remainder_two_power_200_minus_3_mod_7_l205_20576

theorem remainder_two_power_200_minus_3_mod_7 : 
  (2^200 - 3) % 7 = 1 := by sorry

end remainder_two_power_200_minus_3_mod_7_l205_20576


namespace inconsistent_means_l205_20529

theorem inconsistent_means : ¬ ∃ x : ℝ,
  (x + 42 + 78 + 104) / 4 = 62 ∧
  (48 + 62 + 98 + 124 + x) / 5 = 78 := by
  sorry

end inconsistent_means_l205_20529


namespace bees_after_seven_days_l205_20506

/-- Calculates the total number of bees in a hive after a given number of days -/
def total_bees_in_hive (initial_bees : ℕ) (hatch_rate : ℕ) (loss_rate : ℕ) (days : ℕ) : ℕ :=
  initial_bees + days * (hatch_rate - loss_rate) + 1

/-- Theorem stating the total number of bees in the hive after 7 days -/
theorem bees_after_seven_days :
  total_bees_in_hive 12500 3000 900 7 = 27201 := by
  sorry

#eval total_bees_in_hive 12500 3000 900 7

end bees_after_seven_days_l205_20506


namespace med_school_acceptances_l205_20522

theorem med_school_acceptances 
  (total_researched : ℕ) 
  (applied_fraction : ℚ) 
  (accepted_fraction : ℚ) 
  (h1 : total_researched = 42)
  (h2 : applied_fraction = 1 / 3)
  (h3 : accepted_fraction = 1 / 2) :
  ↑⌊(total_researched : ℚ) * applied_fraction * accepted_fraction⌋ = 7 :=
by sorry

end med_school_acceptances_l205_20522


namespace angle_D_value_l205_20557

-- Define the angles
def A : ℝ := 30
def B (D : ℝ) : ℝ := 2 * D
def C (D : ℝ) : ℝ := D + 40

-- Theorem statement
theorem angle_D_value :
  ∀ D : ℝ, A + B D + C D + D = 360 → D = 72.5 := by sorry

end angle_D_value_l205_20557


namespace linear_composition_solution_l205_20521

/-- A linear function f that satisfies f[f(x)] = 4x - 1 -/
def LinearComposition (f : ℝ → ℝ) : Prop :=
  (∃ k b : ℝ, ∀ x, f x = k * x + b) ∧ 
  (∀ x, f (f x) = 4 * x - 1)

/-- The theorem stating that a linear function satisfying the composition condition
    must be one of two specific linear functions -/
theorem linear_composition_solution (f : ℝ → ℝ) (h : LinearComposition f) :
  (∀ x, f x = 2 * x - 1/3) ∨ (∀ x, f x = -2 * x + 1) :=
sorry

end linear_composition_solution_l205_20521


namespace train_crossing_time_l205_20559

theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 300 → 
  train_speed_kmh = 36 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 30 := by
  sorry

end train_crossing_time_l205_20559


namespace total_weight_of_aluminum_carbonate_l205_20584

/-- Atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- Atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Number of Aluminum atoms in Al2(CO3)3 -/
def Al_count : ℕ := 2

/-- Number of Carbon atoms in Al2(CO3)3 -/
def C_count : ℕ := 3

/-- Number of Oxygen atoms in Al2(CO3)3 -/
def O_count : ℕ := 9

/-- Number of moles of Al2(CO3)3 -/
def moles : ℝ := 6

/-- Calculates the molecular weight of Al2(CO3)3 in g/mol -/
def molecular_weight : ℝ :=
  Al_count * Al_weight + C_count * C_weight + O_count * O_weight

/-- Theorem stating the total weight of 6 moles of Al2(CO3)3 -/
theorem total_weight_of_aluminum_carbonate :
  moles * molecular_weight = 1403.94 := by
  sorry

end total_weight_of_aluminum_carbonate_l205_20584


namespace batsman_80_run_innings_l205_20560

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after adding a score -/
def newAverage (stats : BatsmanStats) (score : ℕ) : ℚ :=
  (stats.totalRuns + score) / (stats.innings + 1)

theorem batsman_80_run_innings :
  ∀ (stats : BatsmanStats),
    stats.average = 46 →
    newAverage stats 80 = 48 →
    stats.innings = 16 :=
by sorry

end batsman_80_run_innings_l205_20560


namespace parabola_passes_through_point_l205_20554

/-- The parabola y = (1/2)x^2 - 2 passes through the point (2, 0) -/
theorem parabola_passes_through_point :
  let f : ℝ → ℝ := fun x ↦ (1/2) * x^2 - 2
  f 2 = 0 := by sorry

end parabola_passes_through_point_l205_20554


namespace third_square_is_G_l205_20543

-- Define the set of squares
inductive Square : Type
| A | B | C | D | E | F | G | H

-- Define the placement order
def PlacementOrder : List Square := [Square.F, Square.H, Square.G, Square.D, Square.A, Square.B, Square.C, Square.E]

-- Define the size of each small square
def SmallSquareSize : Nat := 2

-- Define the size of the large square
def LargeSquareSize : Nat := 4

-- Define the total number of squares
def TotalSquares : Nat := 8

-- Define the visibility property
def IsFullyVisible (s : Square) : Prop := s = Square.E

-- Define the third placed square
def ThirdPlacedSquare : Square := PlacementOrder[2]

-- Theorem statement
theorem third_square_is_G :
  (∀ s : Square, s ≠ Square.E → ¬IsFullyVisible s) →
  IsFullyVisible Square.E →
  TotalSquares = 8 →
  SmallSquareSize = 2 →
  LargeSquareSize = 4 →
  ThirdPlacedSquare = Square.G :=
by sorry

end third_square_is_G_l205_20543


namespace exists_special_subset_l205_20593

theorem exists_special_subset : ∃ (X : Set ℤ), ∀ (n : ℤ), ∃! (p : ℤ × ℤ), p.1 ∈ X ∧ p.2 ∈ X ∧ p.1 + 2 * p.2 = n := by
  sorry

end exists_special_subset_l205_20593


namespace number_calculation_l205_20548

theorem number_calculation (x : ℝ) (h : 0.3 * x = 108.0) : x = 360 := by
  sorry

end number_calculation_l205_20548


namespace remainder_17_63_mod_7_l205_20515

theorem remainder_17_63_mod_7 :
  ∃ k : ℤ, 17^63 = 7 * k + 6 :=
by
  sorry

end remainder_17_63_mod_7_l205_20515


namespace probability_all_8_cards_l205_20570

/-- Represents a player in the card game --/
structure Player where
  cards : ℕ

/-- Represents the state of the game --/
structure GameState where
  players : Fin 6 → Player
  cardsDealt : ℕ

/-- The dealing process for a single card --/
def dealCard (state : GameState) : GameState :=
  sorry

/-- The final state after dealing all cards --/
def finalState : GameState :=
  sorry

/-- Checks if all players have exactly 8 cards --/
def allPlayersHave8Cards (state : GameState) : Prop :=
  ∀ i : Fin 6, (state.players i).cards = 8

/-- The probability of all players having 8 cards after dealing --/
def probabilityAllHave8Cards : ℚ :=
  sorry

/-- Theorem stating the probability of all players having 8 cards is 5/6 --/
theorem probability_all_8_cards : probabilityAllHave8Cards = 5/6 :=
  sorry

end probability_all_8_cards_l205_20570


namespace count_numbers_with_seven_is_152_l205_20597

/-- A function that checks if a natural number contains the digit 7 -/
def contains_seven (n : ℕ) : Bool :=
  sorry

/-- The count of natural numbers from 1 to 800 containing the digit 7 -/
def count_numbers_with_seven : ℕ :=
  (List.range 800).filter (λ n => contains_seven (n + 1)) |>.length

/-- Theorem stating that the count of numbers with seven is 152 -/
theorem count_numbers_with_seven_is_152 :
  count_numbers_with_seven = 152 :=
by sorry

end count_numbers_with_seven_is_152_l205_20597


namespace max_whole_nine_one_number_l205_20505

def is_whole_nine_one_number (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 4 ∧
  1 ≤ b ∧ b ≤ 9 ∧
  1 ≤ c ∧ c ≤ 9 ∧
  1 ≤ d ∧ d ≤ 9 ∧
  a + c = 9 ∧
  b - d = 1 ∧
  ∃ k : ℕ, k * (2 * b + c) = 4 * a + 2 * d

def M (a b c d : ℕ) : ℕ :=
  2000 * a + 100 * b + 10 * c + d

theorem max_whole_nine_one_number :
  ∀ a b c d : ℕ,
    is_whole_nine_one_number a b c d →
    M a b c d ≤ 7524 :=
  sorry

end max_whole_nine_one_number_l205_20505


namespace second_day_sales_correct_l205_20599

/-- Represents the sales of sportswear in a clothing store over two days -/
structure SportswearSales where
  first_day : ℕ
  second_day : ℕ

/-- Calculates the sales on the second day based on the first day's sales -/
def second_day_sales (m : ℕ) : ℕ := 2 * m - 3

/-- Theorem stating the relationship between first and second day sales -/
theorem second_day_sales_correct (sales : SportswearSales) :
  sales.first_day = m →
  sales.second_day = 2 * sales.first_day - 3 →
  sales.second_day = second_day_sales m :=
by sorry

end second_day_sales_correct_l205_20599


namespace quadratic_no_real_roots_inequality_l205_20544

theorem quadratic_no_real_roots_inequality (a b c : ℝ) :
  ((b + c) * x^2 + (a + c) * x + (a + b) = 0 → False) →
  4 * a * c - b^2 ≤ 3 * a * (a + b + c) :=
by sorry

end quadratic_no_real_roots_inequality_l205_20544


namespace fraction_inequality_l205_20566

theorem fraction_inequality (x : ℝ) : 
  -1 ≤ x ∧ x ≤ 3 →
  (4 * x + 3 > 2 * (8 - 3 * x) ↔ 13 / 10 < x ∧ x ≤ 3) :=
by sorry

end fraction_inequality_l205_20566


namespace birds_flying_away_l205_20598

theorem birds_flying_away (total : ℕ) (remaining : ℕ) : 
  total = 60 → remaining = 8 → 
  ∃ (F : ℚ), F = 1/3 ∧ 
  (1 - 2/3) * (1 - 2/5) * (1 - F) * total = remaining :=
by sorry

end birds_flying_away_l205_20598


namespace savings_calculation_l205_20512

def income_expenditure_ratio : Rat := 10 / 4
def income : ℕ := 19000
def tax_rate : Rat := 15 / 100
def long_term_investment_rate : Rat := 10 / 100
def short_term_investment_rate : Rat := 20 / 100

def calculate_savings (income_expenditure_ratio : Rat) (income : ℕ) (tax_rate : Rat) 
  (long_term_investment_rate : Rat) (short_term_investment_rate : Rat) : ℕ :=
  sorry

theorem savings_calculation :
  calculate_savings income_expenditure_ratio income tax_rate 
    long_term_investment_rate short_term_investment_rate = 11628 := by
  sorry

end savings_calculation_l205_20512


namespace parallel_planes_equidistant_points_l205_20574

-- Define the concept of a plane
def Plane : Type := sorry

-- Define the concept of a point
def Point : Type := sorry

-- Define what it means for a point to be in a plane
def in_plane (p : Point) (α : Plane) : Prop := sorry

-- Define what it means for three points to be non-collinear
def non_collinear (p q r : Point) : Prop := sorry

-- Define what it means for a point to be equidistant from a plane
def equidistant_from_plane (p : Point) (β : Plane) (d : ℝ) : Prop := sorry

-- Define what it means for two planes to be parallel
def parallel_planes (α β : Plane) : Prop := sorry

-- State the theorem
theorem parallel_planes_equidistant_points (α β : Plane) :
  (∃ (p q r : Point) (d : ℝ), 
    in_plane p α ∧ in_plane q α ∧ in_plane r α ∧
    non_collinear p q r ∧
    equidistant_from_plane p β d ∧
    equidistant_from_plane q β d ∧
    equidistant_from_plane r β d) →
  parallel_planes α β :=
sorry

end parallel_planes_equidistant_points_l205_20574
