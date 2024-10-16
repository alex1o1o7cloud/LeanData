import Mathlib

namespace NUMINAMATH_CALUDE_segment_ratio_vector_coefficients_l102_10206

-- Define the vector type
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points C, D, and Q
variable (C D Q : V)

-- Define the condition that Q is on the line segment CD with the given ratio
def on_segment_with_ratio (C D Q : V) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • C + t • D ∧ t = 5 / 8

-- Theorem statement
theorem segment_ratio_vector_coefficients
  (h : on_segment_with_ratio C D Q) :
  ∃ (s v : ℝ), Q = s • C + v • D ∧ s = 5/8 ∧ v = 3/8 :=
sorry

end NUMINAMATH_CALUDE_segment_ratio_vector_coefficients_l102_10206


namespace NUMINAMATH_CALUDE_vending_machine_drinks_l102_10296

def arcade_problem (num_machines : ℕ) (sections_per_machine : ℕ) (drinks_left : ℕ) (drinks_dispensed : ℕ) : Prop :=
  let drinks_per_section : ℕ := drinks_left + drinks_dispensed
  let drinks_per_machine : ℕ := drinks_per_section * sections_per_machine
  let total_drinks : ℕ := drinks_per_machine * num_machines
  total_drinks = 840

theorem vending_machine_drinks :
  arcade_problem 28 6 3 2 := by
  sorry

end NUMINAMATH_CALUDE_vending_machine_drinks_l102_10296


namespace NUMINAMATH_CALUDE_items_after_price_drop_l102_10297

/-- Calculates the number of items that can be purchased after a price drop -/
theorem items_after_price_drop (original_price : ℚ) (original_quantity : ℕ) (new_price : ℚ) :
  original_price > 0 →
  new_price > 0 →
  new_price < original_price →
  (original_price * original_quantity) / new_price = 20 :=
by
  sorry

#check items_after_price_drop (4 : ℚ) 15 (3 : ℚ)

end NUMINAMATH_CALUDE_items_after_price_drop_l102_10297


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l102_10201

theorem arithmetic_calculations :
  ((-53 + 21 + 79 - 37) = 10) ∧
  ((-9 - 1/3 - (abs (-4 - 5/6)) + (abs (0 - 5 - 1/6)) - 2/3) = -29/3) ∧
  ((-2^3 * (-4)^2 / (4/3) + abs (5 - 8)) = -93) ∧
  ((1/2 + 5/6 - 7/12) / (-1/36) = -27) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l102_10201


namespace NUMINAMATH_CALUDE_distance_to_tangent_point_l102_10221

/-- Two externally tangent circles with a common external tangent -/
structure TangentCircles where
  /-- Radius of the larger circle -/
  r₁ : ℝ
  /-- Radius of the smaller circle -/
  r₂ : ℝ
  /-- The circles are externally tangent -/
  tangent : r₁ > 0 ∧ r₂ > 0
  /-- The common external tangent exists -/
  common_tangent_exists : True

/-- The distance from the center of the larger circle to the point where 
    the common external tangent touches the smaller circle -/
theorem distance_to_tangent_point (c : TangentCircles) (h₁ : c.r₁ = 10) (h₂ : c.r₂ = 5) :
  ∃ d : ℝ, d = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_tangent_point_l102_10221


namespace NUMINAMATH_CALUDE_sport_water_amount_l102_10245

/-- Represents the ratios in a flavored drink formulation -/
structure DrinkFormulation where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation of the drink -/
def standard : DrinkFormulation := ⟨1, 12, 30⟩

/-- The sport formulation of the drink -/
def sport : DrinkFormulation :=
  ⟨standard.flavoring, standard.corn_syrup / 3, standard.water * 2⟩

theorem sport_water_amount (corn_syrup_amount : ℚ) :
  corn_syrup_amount = 1 →
  (corn_syrup_amount * sport.water / sport.corn_syrup) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sport_water_amount_l102_10245


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l102_10298

theorem sum_of_roots_quadratic (x : ℝ) : (x + 3) * (x - 4) = 20 → ∃ y : ℝ, (y + 3) * (y - 4) = 20 ∧ x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l102_10298


namespace NUMINAMATH_CALUDE_f_plus_g_positive_implies_m_bound_l102_10289

/-- The base of the natural logarithm -/
noncomputable def e : ℝ := Real.exp 1

/-- The function f(x) = e^x / x -/
noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

/-- The function g(x) = mx -/
def g (m : ℝ) (x : ℝ) : ℝ := m * x

/-- Theorem stating that if f(x) + g(x) > 0 for all x > 0, then m > -e^2/4 -/
theorem f_plus_g_positive_implies_m_bound (m : ℝ) :
  (∀ x : ℝ, x > 0 → f x + g m x > 0) →
  m > -(e^2 / 4) := by
  sorry

end NUMINAMATH_CALUDE_f_plus_g_positive_implies_m_bound_l102_10289


namespace NUMINAMATH_CALUDE_unique_prime_square_solution_l102_10204

theorem unique_prime_square_solution :
  ∀ (p m : ℕ), 
    Prime p → 
    m > 0 → 
    2 * p^2 + p + 9 = m^2 → 
    p = 5 ∧ m = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_square_solution_l102_10204


namespace NUMINAMATH_CALUDE_tangent_product_l102_10271

theorem tangent_product (α β : Real) (h : α + β = 3 * Real.pi / 4) :
  (1 - Real.tan α) * (1 - Real.tan β) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_l102_10271


namespace NUMINAMATH_CALUDE_f_triangle_condition_l102_10277

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^4 - 4*x + m

-- Define the interval [0, 2]
def I : Set ℝ := Set.Icc 0 2

-- Define the triangle existence condition
def triangle_exists (m : ℝ) : Prop :=
  ∃ (a b c : ℝ), a ∈ I ∧ b ∈ I ∧ c ∈ I ∧
    f m a + f m b > f m c ∧
    f m b + f m c > f m a ∧
    f m c + f m a > f m b

-- State the theorem
theorem f_triangle_condition (m : ℝ) :
  triangle_exists m → m > 14 := by sorry

end NUMINAMATH_CALUDE_f_triangle_condition_l102_10277


namespace NUMINAMATH_CALUDE_player1_always_wins_l102_10200

/-- Represents a card with a number from 1 to 2002 -/
structure Card where
  number : Nat
  h : 1 ≤ number ∧ number ≤ 2002

/-- The game state, including the deck and players' hands -/
structure GameState where
  deck : List Card
  player1_hand : List Card
  player2_hand : List Card

/-- The sum of the last digits of a player's cards -/
def lastDigitSum (hand : List Card) : Nat :=
  (hand.map (λ c => c.number % 10)).sum % 10

/-- Player 1's strategy function -/
def player1Strategy (state : GameState) : Option Card :=
  sorry

/-- Player 2's strategy function (can be any valid strategy) -/
def player2Strategy (state : GameState) : Option Card :=
  sorry

/-- Simulates the game and returns the final state -/
def playGame (initialState : GameState) : GameState :=
  sorry

/-- Theorem stating that Player 1 can always win -/
theorem player1_always_wins :
  ∀ (initialState : GameState),
    initialState.deck.length = 2002 ∧
    (∀ c ∈ initialState.deck, 1 ≤ c.number ∧ c.number ≤ 2002) →
    let finalState := playGame initialState
    lastDigitSum finalState.player1_hand > lastDigitSum finalState.player2_hand :=
  sorry

end NUMINAMATH_CALUDE_player1_always_wins_l102_10200


namespace NUMINAMATH_CALUDE_train_speed_calculation_l102_10202

/-- Given two trains A and B moving towards each other, calculate the speed of train B. -/
theorem train_speed_calculation (length_A length_B : ℝ) (speed_A : ℝ) (crossing_time : ℝ) 
  (h1 : length_A = 225) 
  (h2 : length_B = 150) 
  (h3 : speed_A = 54) 
  (h4 : crossing_time = 15) : 
  (((length_A + length_B) / crossing_time) * (3600 / 1000) - speed_A) = 36 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l102_10202


namespace NUMINAMATH_CALUDE_keegan_class_count_l102_10220

/-- Calculates the number of classes Keegan is taking given his school schedule --/
theorem keegan_class_count :
  ∀ (total_school_time : ℝ) 
    (history_chem_time : ℝ) 
    (avg_other_class_time : ℝ),
  total_school_time = 7.5 →
  history_chem_time = 1.5 →
  avg_other_class_time = 72 / 60 →
  (total_school_time - history_chem_time) / avg_other_class_time + 2 = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_keegan_class_count_l102_10220


namespace NUMINAMATH_CALUDE_books_from_first_shop_l102_10278

theorem books_from_first_shop :
  ∀ (x : ℕ),
    (1000 : ℝ) + 800 = 20 * (x + 40) →
    x = 50 := by
  sorry

end NUMINAMATH_CALUDE_books_from_first_shop_l102_10278


namespace NUMINAMATH_CALUDE_combined_gold_cost_l102_10248

/-- The cost of Gary and Anna's combined gold -/
theorem combined_gold_cost (gary_grams anna_grams : ℕ) (gary_price anna_price : ℚ) : 
  gary_grams = 30 → 
  gary_price = 15 → 
  anna_grams = 50 → 
  anna_price = 20 → 
  gary_grams * gary_price + anna_grams * anna_price = 1450 := by
  sorry


end NUMINAMATH_CALUDE_combined_gold_cost_l102_10248


namespace NUMINAMATH_CALUDE_cone_volume_ratio_l102_10236

/-- Given two sectors of a circle with central angles in the ratio 3:4, 
    the ratio of the volumes of the cones formed by rolling these sectors is 27:64 -/
theorem cone_volume_ratio (r : ℝ) (θ : ℝ) (h₁ h₂ : ℝ) :
  r > 0 → θ > 0 →
  3 * θ + 4 * θ = 2 * π →
  h₁ = Real.sqrt (r^2 - (3 * θ * r / (2 * π))^2) →
  h₂ = Real.sqrt (r^2 - (4 * θ * r / (2 * π))^2) →
  (1/3 * π * (3 * θ * r / (2 * π))^2 * h₁) / (1/3 * π * (4 * θ * r / (2 * π))^2 * h₂) = 27/64 :=
by sorry

#check cone_volume_ratio

end NUMINAMATH_CALUDE_cone_volume_ratio_l102_10236


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l102_10224

/-- The ratio of the area to the square of the perimeter for an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio : 
  let side_length : ℝ := 10
  let perimeter : ℝ := 3 * side_length
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  area / (perimeter ^ 2) = Real.sqrt 3 / 36 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l102_10224


namespace NUMINAMATH_CALUDE_multiply_decimals_l102_10263

theorem multiply_decimals : 0.9 * 0.007 = 0.0063 := by
  sorry

end NUMINAMATH_CALUDE_multiply_decimals_l102_10263


namespace NUMINAMATH_CALUDE_exponent_equation_l102_10209

theorem exponent_equation (a : ℝ) (m : ℝ) (h1 : a ≠ 0) (h2 : a^5 * (a^m)^3 = a^11) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_l102_10209


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l102_10251

/-- The polynomial x^3 - 8x^2 + 17x - 14 -/
def polynomial (x : ℝ) : ℝ := x^3 - 8*x^2 + 17*x - 14

/-- The sum of the kth powers of the roots -/
def s (k : ℕ) : ℝ := sorry

/-- The relation between consecutive s_k values -/
def relation (a b c : ℝ) : Prop :=
  ∀ k : ℕ, k ≥ 1 → s (k+1) = a * s k + b * s (k-1) + c * s (k-2)

theorem sum_of_coefficients :
  ∃ (a b c : ℝ),
    s 0 = 3 ∧ s 1 = 8 ∧ s 2 = 17 ∧
    relation a b c ∧
    a + b + c = 9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l102_10251


namespace NUMINAMATH_CALUDE_function_minimum_condition_l102_10285

/-- A function f(x) = x^2 - 2ax + a has a minimum value in the interval (-∞, 1) if and only if a < 1 -/
theorem function_minimum_condition (a : ℝ) : 
  (∃ (x₀ : ℝ), x₀ < 1 ∧ ∀ (x : ℝ), x < 1 → (x^2 - 2*a*x + a) ≥ (x₀^2 - 2*a*x₀ + a)) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_condition_l102_10285


namespace NUMINAMATH_CALUDE_messenger_speed_l102_10237

/-- Messenger speed problem -/
theorem messenger_speed (team_length : ℝ) (team_speed : ℝ) (total_time : ℝ) :
  team_length = 6 →
  team_speed = 5 →
  total_time = 0.5 →
  ∃ messenger_speed : ℝ,
    messenger_speed > 0 ∧
    (team_length / (messenger_speed + team_speed) + team_length / (messenger_speed - team_speed) = total_time) ∧
    messenger_speed = 25 := by
  sorry

end NUMINAMATH_CALUDE_messenger_speed_l102_10237


namespace NUMINAMATH_CALUDE_jerry_action_figures_l102_10228

theorem jerry_action_figures (total_needed : ℕ) (cost_per_figure : ℕ) (amount_needed : ℕ) :
  total_needed = 16 →
  cost_per_figure = 8 →
  amount_needed = 72 →
  total_needed - (amount_needed / cost_per_figure) = 7 :=
by sorry

end NUMINAMATH_CALUDE_jerry_action_figures_l102_10228


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l102_10257

/-- Two 2D vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (-1, 2*m + 1)
  parallel a b → m = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l102_10257


namespace NUMINAMATH_CALUDE_shirt_sale_tax_percentage_l102_10253

theorem shirt_sale_tax_percentage : 
  let num_fandoms : ℕ := 4
  let shirts_per_fandom : ℕ := 5
  let original_price : ℚ := 15
  let discount_percentage : ℚ := 20 / 100
  let total_paid : ℚ := 264

  let discounted_price : ℚ := original_price * (1 - discount_percentage)
  let total_shirts : ℕ := num_fandoms * shirts_per_fandom
  let total_cost_before_tax : ℚ := discounted_price * total_shirts
  let tax_amount : ℚ := total_paid - total_cost_before_tax
  let tax_percentage : ℚ := tax_amount / total_cost_before_tax * 100

  tax_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_shirt_sale_tax_percentage_l102_10253


namespace NUMINAMATH_CALUDE_choose_three_from_fifteen_l102_10238

theorem choose_three_from_fifteen (n k : ℕ) : n = 15 ∧ k = 3 → Nat.choose n k = 455 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_fifteen_l102_10238


namespace NUMINAMATH_CALUDE_bob_cannot_win_bob_must_choose_nine_l102_10262

/-- Represents the possible game numbers -/
inductive GameNumber
| nineteen : GameNumber
| twenty : GameNumber

/-- Represents the possible starting numbers -/
inductive StartNumber
| nine : StartNumber
| ten : StartNumber

/-- Represents a player in the game -/
inductive Player
| alice : Player
| bob : Player

/-- Represents the state of the game after each turn -/
structure GameState where
  current_sum : ℕ
  current_player : Player

/-- Represents the outcome of the game -/
inductive GameOutcome
| alice_wins : GameOutcome
| bob_wins : GameOutcome
| draw : GameOutcome

/-- Simulates a single turn of the game -/
def play_turn (state : GameState) (alice_number : GameNumber) (bob_number : GameNumber) : GameState :=
  sorry

/-- Simulates the entire game until completion -/
def play_game (start : StartNumber) (alice_number : GameNumber) (bob_number : GameNumber) : GameOutcome :=
  sorry

/-- Theorem stating that Bob cannot win -/
theorem bob_cannot_win :
  ∀ (start : StartNumber) (alice_number bob_number : GameNumber),
    play_game start alice_number bob_number ≠ GameOutcome.bob_wins :=
  sorry

/-- Theorem stating that Bob must choose 9 to prevent Alice from winning -/
theorem bob_must_choose_nine :
  (∀ (alice_number bob_number : GameNumber),
    play_game StartNumber.nine alice_number bob_number ≠ GameOutcome.alice_wins) ∧
  (∃ (alice_number bob_number : GameNumber),
    play_game StartNumber.ten alice_number bob_number = GameOutcome.alice_wins) :=
  sorry

end NUMINAMATH_CALUDE_bob_cannot_win_bob_must_choose_nine_l102_10262


namespace NUMINAMATH_CALUDE_complex_equation_solution_l102_10241

theorem complex_equation_solution (a b : ℝ) 
  (h : (a - 1 : ℂ) + a * I = 3 + 2 * b * I) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l102_10241


namespace NUMINAMATH_CALUDE_product_of_exponents_l102_10272

theorem product_of_exponents (p r s : ℕ) : 
  3^p + 3^5 = 270 →
  2^r + 46 = 78 →
  6^s + 5^4 = 1921 →
  p * r * s = 60 := by
  sorry

end NUMINAMATH_CALUDE_product_of_exponents_l102_10272


namespace NUMINAMATH_CALUDE_extended_segment_vector_representation_l102_10266

/-- Given a line segment AB extended past B to Q with AQ:QB = 7:2,
    prove that Q = (2/9)A + (7/9)B -/
theorem extended_segment_vector_representation 
  (A B Q : ℝ × ℝ) -- Points in 2D plane
  (h : (dist A Q) / (dist Q B) = 7 / 2) -- AQ:QB = 7:2
  : ∃ (x y : ℝ), x = 2/9 ∧ y = 7/9 ∧ Q = x • A + y • B :=
by sorry


end NUMINAMATH_CALUDE_extended_segment_vector_representation_l102_10266


namespace NUMINAMATH_CALUDE_stating_distinguishable_triangles_l102_10291

/-- Represents the number of colors available -/
def num_colors : ℕ := 8

/-- Represents the number of small triangles in the large triangle -/
def num_triangles : ℕ := 4

/-- 
Calculates the number of ways to color a large equilateral triangle 
made of 4 smaller triangles using 8 colors, where no adjacent triangles 
can have the same color.
-/
def count_colorings : ℕ := 
  num_colors * (num_colors - 1) * (num_colors - 2) * (num_colors - 3)

/-- 
Theorem stating that the number of distinguishable large equilateral triangles 
is equal to 1680.
-/
theorem distinguishable_triangles : count_colorings = 1680 := by
  sorry

end NUMINAMATH_CALUDE_stating_distinguishable_triangles_l102_10291


namespace NUMINAMATH_CALUDE_trigonometric_product_equality_l102_10254

theorem trigonometric_product_equality : 
  3.420 * Real.sin (10 * π / 180) * Real.sin (20 * π / 180) * Real.sin (30 * π / 180) * 
  Real.sin (40 * π / 180) * Real.sin (50 * π / 180) * Real.sin (60 * π / 180) * 
  Real.sin (70 * π / 180) * Real.sin (80 * π / 180) = 3 / 256 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equality_l102_10254


namespace NUMINAMATH_CALUDE_parabola_focus_l102_10219

/-- A parabola is defined by the equation x^2 = -8y -/
def parabola (x y : ℝ) : Prop := x^2 = -8*y

/-- The focus of a parabola is a point on its axis of symmetry -/
def is_focus (x y : ℝ) (p : ℝ → ℝ → Prop) : Prop :=
  ∀ (u v : ℝ), p u v → (x = 0 ∧ y = -2)

/-- Theorem: The focus of the parabola x^2 = -8y is located at (0, -2) -/
theorem parabola_focus :
  is_focus 0 (-2) parabola :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l102_10219


namespace NUMINAMATH_CALUDE_p_is_8x_squared_minus_8_l102_10261

-- Define the numerator polynomial
def num (x : ℝ) : ℝ := x^4 - 2*x^3 - 7*x + 6

-- Define the properties of p(x)
def has_vertical_asymptotes (p : ℝ → ℝ) : Prop :=
  p 1 = 0 ∧ p (-1) = 0

def no_horizontal_asymptote (p : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, ∀ x : ℝ, ∃ c : ℝ, |p x| ≤ c * |x|^n

-- Main theorem
theorem p_is_8x_squared_minus_8 (p : ℝ → ℝ) :
  has_vertical_asymptotes p →
  no_horizontal_asymptote p →
  p 2 = 24 →
  ∀ x : ℝ, p x = 8*x^2 - 8 :=
by sorry

end NUMINAMATH_CALUDE_p_is_8x_squared_minus_8_l102_10261


namespace NUMINAMATH_CALUDE_simplify_fourth_root_l102_10244

theorem simplify_fourth_root (a : ℝ) (h : a < 1/2) : 
  (2*a - 1)^2^(1/4) = Real.sqrt (1 - 2*a) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fourth_root_l102_10244


namespace NUMINAMATH_CALUDE_problem_solution_l102_10288

theorem problem_solution (x y z : ℚ) (w : ℚ) : 
  x = (1 / 3) * y → 
  y = (1 / 4) * z → 
  z = 80 → 
  x = 20 / 3 ∧ w = x + y + z ∧ w = 320 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l102_10288


namespace NUMINAMATH_CALUDE_same_type_as_reference_l102_10256

-- Define the type of polynomial expressions
def PolynomialExpr (α : Type) := List (α × ℕ)

-- Function to get the type of a polynomial expression
def exprType (expr : PolynomialExpr ℚ) : PolynomialExpr ℚ :=
  expr.map (λ (c, e) ↦ (1, e))

-- Define the reference expression 3a²b
def reference : PolynomialExpr ℚ := [(3, 2), (1, 1)]

-- Define the given expressions
def expr1 : PolynomialExpr ℚ := [(-2, 2), (1, 1)]  -- -2a²b
def expr2 : PolynomialExpr ℚ := [(-2, 1), (1, 1)]  -- -2ab
def expr3 : PolynomialExpr ℚ := [(2, 1), (2, 1)]   -- 2ab²
def expr4 : PolynomialExpr ℚ := [(2, 2)]           -- 2a²

theorem same_type_as_reference :
  (exprType expr1 = exprType reference) ∧
  (exprType expr2 ≠ exprType reference) ∧
  (exprType expr3 ≠ exprType reference) ∧
  (exprType expr4 ≠ exprType reference) :=
by sorry

end NUMINAMATH_CALUDE_same_type_as_reference_l102_10256


namespace NUMINAMATH_CALUDE_digit_sum_problem_l102_10240

theorem digit_sum_problem (x y z w : ℕ) : 
  (x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w) →
  (x < 10 ∧ y < 10 ∧ z < 10 ∧ w < 10) →
  (100 * x + 10 * y + z) + (100 * w + 10 * z + x) = 1000 →
  z + x ≥ 10 →
  y + z < 10 →
  x + y + z + w = 19 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l102_10240


namespace NUMINAMATH_CALUDE_williams_tickets_l102_10270

/-- William's ticket problem -/
theorem williams_tickets : 
  ∀ (initial_tickets additional_tickets : ℕ),
  initial_tickets = 15 → 
  additional_tickets = 3 → 
  initial_tickets + additional_tickets = 18 := by
sorry

end NUMINAMATH_CALUDE_williams_tickets_l102_10270


namespace NUMINAMATH_CALUDE_quadratic_root_one_iff_sum_zero_l102_10295

theorem quadratic_root_one_iff_sum_zero (a b c : ℝ) :
  (∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = 1) ↔ a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_one_iff_sum_zero_l102_10295


namespace NUMINAMATH_CALUDE_impossible_score_l102_10210

/-- Represents the score of a quiz -/
structure QuizScore where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ
  total_questions : ℕ
  score : ℤ

/-- The quiz scoring system -/
def quiz_score (qs : QuizScore) : Prop :=
  qs.correct + qs.unanswered + qs.incorrect = qs.total_questions ∧
  qs.score = 5 * qs.correct + 2 * qs.unanswered - qs.incorrect

theorem impossible_score : 
  ∀ qs : QuizScore, 
  qs.total_questions = 25 → 
  quiz_score qs → 
  qs.score ≠ 127 := by
sorry

end NUMINAMATH_CALUDE_impossible_score_l102_10210


namespace NUMINAMATH_CALUDE_parallel_heaters_boiling_time_l102_10247

/-- Given two heaters connected to the same direct current source,
    prove that the time to boil water when connected in parallel
    is (t₁ * t₂) / (t₁ + t₂), where t₁ and t₂ are the times taken
    by each heater individually. -/
theorem parallel_heaters_boiling_time
  (t₁ t₂ : ℝ)
  (h₁ : t₁ > 0)
  (h₂ : t₂ > 0)
  (boil_time : ℝ → ℝ → ℝ) :
  boil_time t₁ t₂ = t₁ * t₂ / (t₁ + t₂) :=
by sorry

end NUMINAMATH_CALUDE_parallel_heaters_boiling_time_l102_10247


namespace NUMINAMATH_CALUDE_amanda_ticket_sales_l102_10252

/-- The number of tickets Amanda sells on the first day -/
def day1_tickets : ℕ := 5 * 4

/-- The number of tickets Amanda sells on the second day -/
def day2_tickets : ℕ := 32

/-- The number of tickets Amanda needs to sell on the third day -/
def day3_tickets : ℕ := 28

/-- The total number of tickets Amanda needs to sell -/
def total_tickets : ℕ := day1_tickets + day2_tickets + day3_tickets

/-- Theorem stating that the total number of tickets Amanda needs to sell is 80 -/
theorem amanda_ticket_sales : total_tickets = 80 := by
  sorry

end NUMINAMATH_CALUDE_amanda_ticket_sales_l102_10252


namespace NUMINAMATH_CALUDE_inequality_solution_set_l102_10214

/-- The solution set of the inequality (a^2-1)x^2-(a-1)x-1 < 0 is ℝ if and only if -3/5 < a ≤ 1 -/
theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ -3/5 < a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l102_10214


namespace NUMINAMATH_CALUDE_problem_solution_l102_10273

-- Define the variables
variable (a b c : ℝ)

-- Define the conditions
def condition1 : Prop := (5 * a + 2) ^ (1/3 : ℝ) = 3
def condition2 : Prop := (3 * a + b - 1) ^ (1/2 : ℝ) = 4
def condition3 : Prop := c = ⌊Real.sqrt 13⌋

-- Define the theorem
theorem problem_solution (h1 : condition1 a) (h2 : condition2 a b) (h3 : condition3 c) :
  a = 5 ∧ b = 2 ∧ c = 3 ∧ (3 * a - b + c) ^ (1/2 : ℝ) = 4 ∨ (3 * a - b + c) ^ (1/2 : ℝ) = -4 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l102_10273


namespace NUMINAMATH_CALUDE_pie_chart_proportions_l102_10264

theorem pie_chart_proportions :
  ∀ (white black gray blue : ℚ),
    white = 3 * black →
    black = 2 * gray →
    blue = gray →
    white + black + gray + blue = 1 →
    white = 3/5 ∧ black = 1/5 ∧ gray = 1/10 ∧ blue = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_pie_chart_proportions_l102_10264


namespace NUMINAMATH_CALUDE_rectangle_with_hole_area_l102_10290

theorem rectangle_with_hole_area (x : ℝ) :
  let large_length : ℝ := 2*x + 9
  let large_width : ℝ := x + 6
  let hole_side : ℝ := x - 1
  let large_area : ℝ := large_length * large_width
  let hole_area : ℝ := hole_side * hole_side
  large_area - hole_area = x^2 + 23*x + 53 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_with_hole_area_l102_10290


namespace NUMINAMATH_CALUDE_triangle_construction_l102_10215

-- Define the necessary structures
structure Line where
  -- Add necessary fields for a line

structure Point where
  -- Add necessary fields for a point

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the necessary functions
def is_on_line (p : Point) (l : Line) : Prop :=
  sorry

def is_foot_of_altitude (m : Point) (v : Point) (s : Point) (t : Point) : Prop :=
  sorry

-- Main theorem
theorem triangle_construction (L : Line) (M₁ M₂ : Point) :
  ∃ (ABC A'B'C' : Triangle),
    (is_on_line ABC.C L ∧ is_on_line ABC.B L) ∧
    (is_on_line A'B'C'.C L ∧ is_on_line A'B'C'.B L) ∧
    (is_foot_of_altitude M₁ ABC.A ABC.B ABC.C) ∧
    (is_foot_of_altitude M₂ ABC.B ABC.A ABC.C) ∧
    (is_foot_of_altitude M₁ A'B'C'.A A'B'C'.B A'B'C'.C) ∧
    (is_foot_of_altitude M₂ A'B'C'.B A'B'C'.A A'B'C'.C) :=
  sorry


end NUMINAMATH_CALUDE_triangle_construction_l102_10215


namespace NUMINAMATH_CALUDE_polynomial_coefficient_b_l102_10242

theorem polynomial_coefficient_b (a b c : ℚ) :
  (∀ x : ℚ, (3 * x^2 - 2 * x + 5/4) * (a * x^2 + b * x + c) = 
    9 * x^4 - 5 * x^3 + 31/4 * x^2 - 10/3 * x + 5/12) →
  b = 1/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_b_l102_10242


namespace NUMINAMATH_CALUDE_sum_equation_implies_n_value_l102_10216

theorem sum_equation_implies_n_value : 
  990 + 992 + 994 + 996 + 998 = 5000 - N → N = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_equation_implies_n_value_l102_10216


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l102_10232

theorem arithmetic_expression_equality : 7 ^ 8 - 6 / 2 + 9 ^ 3 + 3 + 12 = 5765542 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l102_10232


namespace NUMINAMATH_CALUDE_number_of_teams_l102_10281

/-- The number of teams in the league -/
def n : ℕ := sorry

/-- The total number of games played in the season -/
def total_games : ℕ := 4900

/-- Each team faces every other team this many times -/
def games_per_pair : ℕ := 4

theorem number_of_teams : 
  (n * games_per_pair * (n - 1)) / 2 = total_games ∧ n = 50 := by sorry

end NUMINAMATH_CALUDE_number_of_teams_l102_10281


namespace NUMINAMATH_CALUDE_triangle_vector_relation_l102_10286

-- Define the triangle ABC and vectors a and b
variable (A B C : EuclideanSpace ℝ (Fin 2))
variable (a b : EuclideanSpace ℝ (Fin 2))

-- Define points P and Q
variable (P : EuclideanSpace ℝ (Fin 2))
variable (Q : EuclideanSpace ℝ (Fin 2))

-- State the theorem
theorem triangle_vector_relation
  (h1 : B - A = a)
  (h2 : C - A = b)
  (h3 : P - A = (1/3) • (B - A))
  (h4 : Q - B = (1/3) • (C - B)) :
  Q - P = (1/3) • a + (1/3) • b := by sorry

end NUMINAMATH_CALUDE_triangle_vector_relation_l102_10286


namespace NUMINAMATH_CALUDE_decode_1236_is_rand_l102_10225

/-- Represents a coding scheme for words -/
structure CodeScheme where
  range_code : String
  random_code : String

/-- Decodes a given code based on the coding scheme -/
def decode (scheme : CodeScheme) (code : String) : String :=
  sorry

/-- The specific coding scheme used in the problem -/
def problem_scheme : CodeScheme :=
  { range_code := "12345", random_code := "123678" }

/-- The theorem stating that 1236 decodes to "rand" under the given scheme -/
theorem decode_1236_is_rand :
  decode problem_scheme "1236" = "rand" :=
sorry

end NUMINAMATH_CALUDE_decode_1236_is_rand_l102_10225


namespace NUMINAMATH_CALUDE_find_number_l102_10249

theorem find_number : ∃ x : ℕ, x * 9999 = 724777430 ∧ x = 72483 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l102_10249


namespace NUMINAMATH_CALUDE_share_ratio_proof_l102_10260

theorem share_ratio_proof (total amount : ℕ) (a_share b_share c_share : ℕ) 
  (h1 : amount = 595)
  (h2 : a_share + b_share + c_share = amount)
  (h3 : a_share = 420)
  (h4 : b_share = 105)
  (h5 : c_share = 70)
  (h6 : 3 * a_share = 2 * b_share) : 
  b_share * 2 = c_share * 3 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_proof_l102_10260


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l102_10274

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 5}

theorem complement_intersection_theorem :
  (A ∩ B)ᶜ = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l102_10274


namespace NUMINAMATH_CALUDE_quadratic_has_two_distinct_roots_find_k_value_l102_10282

/-- A quadratic equation with parameter k -/
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 + (2*k - 1)*x - k - 2

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ := (2*k - 1)^2 - 4*1*(-k - 2)

theorem quadratic_has_two_distinct_roots (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 :=
sorry

theorem find_k_value (k : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : quadratic k x₁ = 0)
  (h₂ : quadratic k x₂ = 0)
  (h₃ : x₁ + x₂ - 4*x₁*x₂ = 1) :
  k = -4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_two_distinct_roots_find_k_value_l102_10282


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l102_10239

theorem parallelogram_side_length (s : ℝ) : 
  s > 0 → -- side length is positive
  let angle : ℝ := 30 * π / 180 -- 30 degrees in radians
  let area : ℝ := 12 * Real.sqrt 3 -- area of the parallelogram
  s * (s * Real.sin angle) = area → -- area formula for parallelogram
  s = 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l102_10239


namespace NUMINAMATH_CALUDE_expression_evaluation_l102_10279

theorem expression_evaluation : 
  |-2| + (1/4 : ℝ) - 1 - 4 * Real.cos (π/4) + Real.sqrt 8 = 5/4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l102_10279


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_l102_10292

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_factorials : ℕ → ℕ
  | 0 => 0
  | n + 1 => factorial (5 * n + 3) + sum_factorials n

theorem last_two_digits_sum_factorials :
  last_two_digits (sum_factorials 20) = 26 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_l102_10292


namespace NUMINAMATH_CALUDE_cube_edge_length_l102_10265

theorem cube_edge_length (x : ℝ) : x > 0 → 6 * x^2 = 1014 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l102_10265


namespace NUMINAMATH_CALUDE_smallest_n_value_l102_10213

/-- The number of ordered quadruplets (a, b, c, d) satisfying the conditions -/
def num_quadruplets : ℕ := 60000

/-- The greatest common divisor of the quadruplets -/
def gcd_value : ℕ := 60

/-- The function that counts the number of ordered quadruplets (a, b, c, d) 
    such that gcd(a, b, c, d) = gcd_value and lcm(a, b, c, d) = n -/
def count_quadruplets (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that 6480 is the smallest value of n 
    satisfying the given conditions -/
theorem smallest_n_value : 
  (∃ n : ℕ, count_quadruplets n = num_quadruplets) →
  (∀ m : ℕ, count_quadruplets m = num_quadruplets → m ≥ 6480) ∧
  (count_quadruplets 6480 = num_quadruplets) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_value_l102_10213


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l102_10259

/-- Represents the number of athletes selected in a stratified sample -/
structure StratifiedSample where
  totalMale : ℕ
  totalFemale : ℕ
  selectedMale : ℕ
  selectedFemale : ℕ

/-- Checks if the sample maintains the same ratio as the total population -/
def isProportionalSample (s : StratifiedSample) : Prop :=
  s.totalMale * s.selectedFemale = s.totalFemale * s.selectedMale

/-- Theorem: Given the conditions, the number of selected female athletes is 6 -/
theorem stratified_sample_theorem (s : StratifiedSample) :
  s.totalMale = 56 →
  s.totalFemale = 42 →
  s.selectedMale = 8 →
  isProportionalSample s →
  s.selectedFemale = 6 := by
  sorry

#check stratified_sample_theorem

end NUMINAMATH_CALUDE_stratified_sample_theorem_l102_10259


namespace NUMINAMATH_CALUDE_scientific_notation_274_million_l102_10235

theorem scientific_notation_274_million :
  274000000 = 2.74 * (10 : ℝ)^8 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_274_million_l102_10235


namespace NUMINAMATH_CALUDE_athleteHitsBullseyeUncertain_l102_10211

-- Define the type for different kinds of events
inductive EventType
  | Certain
  | Impossible
  | Uncertain

-- Define the event
def athleteHitsBullseye : EventType := EventType.Uncertain

-- Theorem statement
theorem athleteHitsBullseyeUncertain : athleteHitsBullseye = EventType.Uncertain := by
  sorry

end NUMINAMATH_CALUDE_athleteHitsBullseyeUncertain_l102_10211


namespace NUMINAMATH_CALUDE_birthday_cards_count_l102_10231

def total_amount_spent : ℕ := 70
def cost_per_card : ℕ := 2
def christmas_cards : ℕ := 20

def total_cards : ℕ := total_amount_spent / cost_per_card

def birthday_cards : ℕ := total_cards - christmas_cards

theorem birthday_cards_count : birthday_cards = 15 := by
  sorry

end NUMINAMATH_CALUDE_birthday_cards_count_l102_10231


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l102_10268

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₄ + a₅ = -31 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l102_10268


namespace NUMINAMATH_CALUDE_student_number_problem_l102_10217

theorem student_number_problem (x : ℝ) : 4 * x - 142 = 110 → x = 63 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l102_10217


namespace NUMINAMATH_CALUDE_point_coordinates_l102_10283

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Determines if a point is in the fourth quadrant -/
def inFourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: If a point P is in the fourth quadrant, its distance to the x-axis is 5,
    and its distance to the y-axis is 3, then its coordinates are (3, -5) -/
theorem point_coordinates (p : Point) 
    (h1 : inFourthQuadrant p)
    (h2 : distanceToXAxis p = 5)
    (h3 : distanceToYAxis p = 3) :
    p = Point.mk 3 (-5) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l102_10283


namespace NUMINAMATH_CALUDE_franks_reading_time_l102_10299

/-- Represents the problem of calculating Frank's effective reading time --/
theorem franks_reading_time (total_pages : ℕ) (reading_speed : ℕ) (total_days : ℕ) :
  total_pages = 2345 →
  reading_speed = 50 →
  total_days = 34 →
  ∃ (effective_time : ℚ),
    effective_time > 2.03 ∧
    effective_time < 2.05 ∧
    effective_time = (total_pages : ℚ) / reading_speed / ((2 * total_days : ℚ) / 3) :=
by sorry

end NUMINAMATH_CALUDE_franks_reading_time_l102_10299


namespace NUMINAMATH_CALUDE_chime_2003_date_l102_10205

/-- Represents a date with year, month, and day -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Represents a time with hour and minute -/
structure Time where
  hour : Nat
  minute : Nat

/-- Calculates the number of chimes for a given hour -/
def hourChimes (hour : Nat) : Nat :=
  hour % 12

/-- Calculates the total number of chimes from a start date and time to an end date -/
def totalChimes (startDate : Date) (startTime : Time) (endDate : Date) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem chime_2003_date :
  let startDate := Date.mk 2003 2 28
  let startTime := Time.mk 15 15
  let endDate := Date.mk 2003 3 22
  totalChimes startDate startTime endDate = 2003 :=
sorry

end NUMINAMATH_CALUDE_chime_2003_date_l102_10205


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l102_10258

-- Define the equations
def equation1 (x : ℝ) : Prop := 4 - x = 3 * (2 - x)
def equation2 (x : ℝ) : Prop := (2 * x - 1) / 2 - (2 * x + 5) / 3 = (6 * x - 1) / 6 - 1

-- Theorem for equation 1
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1 := by sorry

-- Theorem for equation 2
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = -1.5 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l102_10258


namespace NUMINAMATH_CALUDE_distributor_cost_l102_10229

theorem distributor_cost (commission_rate : Real) (profit_rate : Real) (observed_price : Real) :
  commission_rate = 0.20 →
  profit_rate = 0.20 →
  observed_price = 30 →
  ∃ (cost : Real),
    cost = 31.25 ∧
    observed_price = (1 - commission_rate) * (cost * (1 + profit_rate)) :=
by sorry

end NUMINAMATH_CALUDE_distributor_cost_l102_10229


namespace NUMINAMATH_CALUDE_max_distance_ellipse_to_line_l102_10243

/-- An ellipse in the xy-plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → Prop := λ x y ↦ x^2 / (a^2) + y^2 / (b^2) = 1

/-- A line in the xy-plane represented by parametric equations -/
structure ParametricLine where
  fx : ℝ → ℝ
  fy : ℝ → ℝ

/-- The distance between a point and a line -/
def distance_point_to_line (x y : ℝ) (l : ParametricLine) : ℝ := sorry

/-- The maximum distance from a point on an ellipse to a line -/
def max_distance (e : Ellipse) (l : ParametricLine) : ℝ := sorry

theorem max_distance_ellipse_to_line :
  let e : Ellipse := { a := 4, b := 2, equation := λ x y ↦ x^2 / 16 + y^2 / 4 = 1 }
  let l : ParametricLine := { fx := λ t ↦ Real.sqrt 2 - t, fy := λ t ↦ t / 2 }
  max_distance e l = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_max_distance_ellipse_to_line_l102_10243


namespace NUMINAMATH_CALUDE_exists_n_with_totient_inequality_l102_10267

open Nat

theorem exists_n_with_totient_inequality : 
  ∃ (n : ℕ), n > 0 ∧ totient (2*n - 1) + totient (2*n + 1) < (1 : ℚ) / 1000 * totient (2*n) :=
by sorry

end NUMINAMATH_CALUDE_exists_n_with_totient_inequality_l102_10267


namespace NUMINAMATH_CALUDE_sum_of_remaining_segments_l102_10218

/-- Represents a rectangular figure with some interior segments -/
structure RectFigure where
  left : ℝ
  right : ℝ
  bottomLeft : ℝ
  topLeft : ℝ
  topRight : ℝ

/-- Calculates the sum of remaining segments after removing four sides -/
def remainingSum (f : RectFigure) : ℝ :=
  f.left + f.right + (f.bottomLeft + f.topLeft + f.topRight) + f.topRight

/-- Theorem stating that for the given measurements, the sum of remaining segments is 23 -/
theorem sum_of_remaining_segments :
  let f : RectFigure := {
    left := 10,
    right := 7,
    bottomLeft := 3,
    topLeft := 1,
    topRight := 1
  }
  remainingSum f = 23 := by sorry

end NUMINAMATH_CALUDE_sum_of_remaining_segments_l102_10218


namespace NUMINAMATH_CALUDE_sixtieth_number_is_sixteen_l102_10287

/-- Defines the number of elements in each row of the sequence -/
def elementsInRow (n : ℕ) : ℕ := 2 * n

/-- Defines the value of elements in each row of the sequence -/
def valueInRow (n : ℕ) : ℕ := 2 * n

/-- Calculates the cumulative sum of elements up to and including row n -/
def cumulativeSum (n : ℕ) : ℕ :=
  (List.range n).map elementsInRow |>.sum

/-- Finds the row number for a given position in the sequence -/
def findRow (position : ℕ) : ℕ :=
  (List.range position).find? (fun n => cumulativeSum (n + 1) ≥ position)
    |>.getD 0

/-- The main theorem stating that the 60th number in the sequence is 16 -/
theorem sixtieth_number_is_sixteen :
  valueInRow (findRow 60 + 1) = 16 := by
  sorry

#eval valueInRow (findRow 60 + 1)

end NUMINAMATH_CALUDE_sixtieth_number_is_sixteen_l102_10287


namespace NUMINAMATH_CALUDE_same_color_sock_pairs_l102_10208

def num_white_socks : Nat := 5
def num_brown_socks : Nat := 3
def num_blue_socks : Nat := 2
def num_red_socks : Nat := 2

def total_socks : Nat := num_white_socks + num_brown_socks + num_blue_socks + num_red_socks

def choose (n k : Nat) : Nat :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem same_color_sock_pairs : 
  choose num_white_socks 2 + choose num_brown_socks 2 + choose num_blue_socks 2 + choose num_red_socks 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_same_color_sock_pairs_l102_10208


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l102_10212

theorem min_reciprocal_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 3) :
  (1/x + 1/y) ≥ 1 + (2*Real.sqrt 2)/3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 3 ∧ 1/x₀ + 1/y₀ = 1 + (2*Real.sqrt 2)/3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l102_10212


namespace NUMINAMATH_CALUDE_class_average_problem_l102_10246

theorem class_average_problem (total_students : ℕ) (high_scorers : ℕ) (zero_scorers : ℕ) 
  (high_score : ℕ) (class_average : ℚ) :
  total_students = 28 →
  high_scorers = 4 →
  zero_scorers = 3 →
  high_score = 95 →
  class_average = 47.32142857142857 →
  let remaining_students := total_students - high_scorers - zero_scorers
  let total_score := total_students * class_average
  let high_score_total := high_scorers * high_score
  let remaining_score := total_score - high_score_total
  remaining_score / remaining_students = 45 := by
    sorry

#eval (28 : ℚ) * 47.32142857142857 -- To verify the total score

end NUMINAMATH_CALUDE_class_average_problem_l102_10246


namespace NUMINAMATH_CALUDE_square_of_real_not_always_positive_l102_10223

theorem square_of_real_not_always_positive : ¬ (∀ a : ℝ, a^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_of_real_not_always_positive_l102_10223


namespace NUMINAMATH_CALUDE_wrapping_paper_ratio_l102_10275

theorem wrapping_paper_ratio : 
  ∀ (p1 p2 p3 : ℝ),
  p1 = 2 →
  p3 = p1 + p2 →
  p1 + p2 + p3 = 7 →
  p2 / p1 = 3 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_ratio_l102_10275


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l102_10293

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), (n > 0) ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    p₁ * p₂ * p₃ * p₄ = n) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      q₁ * q₂ * q₃ * q₄ = m)) ∧
  n = 210 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l102_10293


namespace NUMINAMATH_CALUDE_point_coordinates_l102_10280

-- Define a point in the coordinate plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the second quadrant
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

-- Define the distance from a point to the x-axis
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

-- Define the distance from a point to the y-axis
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

-- Theorem statement
theorem point_coordinates (p : Point) 
  (h1 : secondQuadrant p)
  (h2 : distanceToXAxis p = 4)
  (h3 : distanceToYAxis p = 5) :
  p.x = -5 ∧ p.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l102_10280


namespace NUMINAMATH_CALUDE_sequence_integer_count_l102_10233

def sequence_term (n : ℕ) : ℚ :=
  15625 / (5 ^ n)

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem sequence_integer_count :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    (∀ (n : ℕ), n ≥ k → ¬ is_integer (sequence_term n))) ∧
  (∀ (m : ℕ), m > 0 →
    ((∀ (n : ℕ), n < m → is_integer (sequence_term n)) ∧
     (∀ (n : ℕ), n ≥ m → ¬ is_integer (sequence_term n)))
    → m = 7) :=
by sorry

end NUMINAMATH_CALUDE_sequence_integer_count_l102_10233


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l102_10269

theorem negative_fractions_comparison : -3/4 > -4/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l102_10269


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l102_10226

/-- For a quadratic equation x^2 + 2x - k = 0 to have two distinct real roots, k must be greater than -1. -/
theorem quadratic_distinct_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x - k = 0 ∧ y^2 + 2*y - k = 0) ↔ k > -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l102_10226


namespace NUMINAMATH_CALUDE_max_difference_reversed_digits_l102_10222

/-- Two-digit positive integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem max_difference_reversed_digits (q r : ℕ) :
  TwoDigitInt q ∧ TwoDigitInt r ∧
  r = reverseDigits q ∧
  (q > r → q - r < 20) ∧
  (r > q → r - q < 20) →
  (q > r → q - r ≤ 18) ∧
  (r > q → r - q ≤ 18) :=
sorry

end NUMINAMATH_CALUDE_max_difference_reversed_digits_l102_10222


namespace NUMINAMATH_CALUDE_power_function_through_point_l102_10250

theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f x = x^a) →  -- f is a power function with exponent a
  f 2 = 16 →              -- f passes through the point (2, 16)
  a = 4 := by             -- prove that a = 4
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l102_10250


namespace NUMINAMATH_CALUDE_first_pipe_fill_time_l102_10255

def cistern_problem (x : ℝ) : Prop :=
  let second_pipe_time : ℝ := 15
  let both_pipes_time : ℝ := 6
  let remaining_time : ℝ := 1.5
  (both_pipes_time / x + both_pipes_time / second_pipe_time + remaining_time / second_pipe_time) = 1

theorem first_pipe_fill_time :
  ∃ x : ℝ, cistern_problem x ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_first_pipe_fill_time_l102_10255


namespace NUMINAMATH_CALUDE_race_participants_l102_10227

/-- Represents a bicycle race with participants. -/
structure BicycleRace where
  participants : ℕ
  petya_position : ℕ
  vasya_position : ℕ
  vasya_position_from_end : ℕ

/-- The bicycle race satisfies the given conditions. -/
def valid_race (race : BicycleRace) : Prop :=
  race.petya_position = 10 ∧
  race.vasya_position = race.petya_position - 1 ∧
  race.vasya_position_from_end = 15

theorem race_participants (race : BicycleRace) :
  valid_race race → race.participants = 23 := by
  sorry

#check race_participants

end NUMINAMATH_CALUDE_race_participants_l102_10227


namespace NUMINAMATH_CALUDE_power_division_l102_10203

theorem power_division (x : ℝ) (h : x ≠ 0) : x^8 / x^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l102_10203


namespace NUMINAMATH_CALUDE_a_less_than_two_necessary_and_sufficient_l102_10294

theorem a_less_than_two_necessary_and_sufficient (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x| > a) ↔ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_two_necessary_and_sufficient_l102_10294


namespace NUMINAMATH_CALUDE_cubic_root_interval_l102_10276

theorem cubic_root_interval (a b : ℤ) : 
  (∃ x : ℝ, x^3 - x + 1 = 0 ∧ a < x ∧ x < b) →
  b - a = 1 →
  a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_interval_l102_10276


namespace NUMINAMATH_CALUDE_system_solution_l102_10284

theorem system_solution (x y : ℝ) : 
  (x + 2*y = 2 ∧ x - 2*y = 6) ↔ (x = 4 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l102_10284


namespace NUMINAMATH_CALUDE_expression_evaluation_l102_10230

theorem expression_evaluation (a b : ℝ) (h1 : a = 1) (h2 : b = -3) :
  (a - b)^2 - 2*a*(a + 3*b) + (a + 2*b)*(a - 2*b) = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l102_10230


namespace NUMINAMATH_CALUDE_mistaken_calculation_l102_10234

theorem mistaken_calculation (x : ℕ) : 
  423 - x = 421 → (423 * x) + 421 - 500 = 767 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_calculation_l102_10234


namespace NUMINAMATH_CALUDE_r_power_four_plus_inverse_l102_10207

theorem r_power_four_plus_inverse (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_power_four_plus_inverse_l102_10207
