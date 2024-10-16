import Mathlib

namespace NUMINAMATH_CALUDE_dogwood_trees_planted_tomorrow_l1446_144649

theorem dogwood_trees_planted_tomorrow (initial_trees : ℕ) (planted_today : ℕ) (final_trees : ℕ) : 
  initial_trees = 7 → planted_today = 5 → final_trees = 16 → 
  final_trees - (initial_trees + planted_today) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_planted_tomorrow_l1446_144649


namespace NUMINAMATH_CALUDE_power_of_fraction_l1446_144666

theorem power_of_fraction : (3 / 4 : ℚ) ^ 3 = 27 / 64 := by sorry

end NUMINAMATH_CALUDE_power_of_fraction_l1446_144666


namespace NUMINAMATH_CALUDE_min_dot_product_on_hyperbola_l1446_144604

/-- The curve C: x^2 - y^2 = 1 (x > 0) -/
def C (x y : ℝ) : Prop := x^2 - y^2 = 1 ∧ x > 0

/-- The dot product function f -/
def f (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ * x₂ + y₁ * y₂

theorem min_dot_product_on_hyperbola :
  ∀ x₁ y₁ x₂ y₂ : ℝ, C x₁ y₁ → C x₂ y₂ → 
  ∃ m : ℝ, m = 1 ∧ ∀ a b c d : ℝ, C a b → C c d → f x₁ y₁ x₂ y₂ ≥ m ∧ f a b c d ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_on_hyperbola_l1446_144604


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l1446_144624

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The symmetric point about the x-axis -/
def symmetricAboutXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem symmetric_point_x_axis :
  let P : Point := { x := -1, y := 2 }
  symmetricAboutXAxis P = { x := -1, y := -2 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l1446_144624


namespace NUMINAMATH_CALUDE_joanna_initial_gumballs_l1446_144669

/-- 
Given:
- Jacques had 60 gumballs initially
- They purchased 4 times their initial total
- After sharing equally, each got 250 gumballs
Prove that Joanna initially had 40 gumballs
-/
theorem joanna_initial_gumballs : 
  ∀ (j : ℕ), -- j represents Joanna's initial number of gumballs
  let jacques_initial := 60
  let total_initial := j + jacques_initial
  let purchased := 4 * total_initial
  let total_final := total_initial + purchased
  let each_after_sharing := 250
  total_final = 2 * each_after_sharing →
  j = 40 := by
sorry

end NUMINAMATH_CALUDE_joanna_initial_gumballs_l1446_144669


namespace NUMINAMATH_CALUDE_optimal_water_tank_design_l1446_144612

/-- Represents the dimensions and costs of a rectangular water tank -/
structure WaterTank where
  volume : ℝ
  depth : ℝ
  bottomCost : ℝ
  wallCost : ℝ

/-- Calculates the total cost of constructing the water tank -/
def totalCost (tank : WaterTank) (length width : ℝ) : ℝ :=
  tank.bottomCost * length * width + 
  tank.wallCost * (2 * length * tank.depth + 2 * width * tank.depth)

/-- Theorem stating the optimal dimensions and minimum cost of the water tank -/
theorem optimal_water_tank_design (tank : WaterTank) 
  (h_volume : tank.volume = 4800)
  (h_depth : tank.depth = 3)
  (h_bottom_cost : tank.bottomCost = 150)
  (h_wall_cost : tank.wallCost = 120) :
  ∃ (cost : ℝ),
    (∀ length width, 
      length * width * tank.depth = tank.volume → 
      totalCost tank length width ≥ cost) ∧
    totalCost tank 40 40 = cost ∧
    cost = 297600 := by
  sorry

end NUMINAMATH_CALUDE_optimal_water_tank_design_l1446_144612


namespace NUMINAMATH_CALUDE_equation_decomposition_l1446_144681

-- Define the equation
def equation (x y : ℝ) : Prop := y^6 - 9*x^6 = 3*y^3 - 1

-- Define a parabola
def is_parabola (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Theorem statement
theorem equation_decomposition :
  ∃ f g : ℝ → ℝ, 
    (∀ x y, equation x y ↔ (y = f x ∨ y = g x)) ∧
    is_parabola f ∧ is_parabola g :=
sorry

end NUMINAMATH_CALUDE_equation_decomposition_l1446_144681


namespace NUMINAMATH_CALUDE_least_sum_m_n_l1446_144652

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (m.val > 0 ∧ n.val > 0) ∧
  (Nat.gcd (m.val + n.val) 210 = 1) ∧
  (∃ (k : ℕ), m.val ^ m.val = k * (n.val ^ n.val)) ∧
  (¬ ∃ (l : ℕ), m.val = l * n.val) ∧
  (m.val + n.val = 407) ∧
  (∀ (p q : ℕ+), 
    (p.val > 0 ∧ q.val > 0) →
    (Nat.gcd (p.val + q.val) 210 = 1) →
    (∃ (k : ℕ), p.val ^ p.val = k * (q.val ^ q.val)) →
    (¬ ∃ (l : ℕ), p.val = l * q.val) →
    (p.val + q.val ≥ 407)) :=
by sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l1446_144652


namespace NUMINAMATH_CALUDE_horner_rule_v3_equals_18_horner_rule_correctness_main_theorem_l1446_144627

/-- Horner's Rule for a specific polynomial -/
def horner_v3 (x : ℝ) : ℝ := ((x + 3) * x - 1) * x

/-- The polynomial f(x) = x^5 + 3x^4 - x^3 + 2x - 1 -/
def f (x : ℝ) : ℝ := x^5 + 3*x^4 - x^3 + 2*x - 1

theorem horner_rule_v3_equals_18 :
  horner_v3 2 = 18 := by sorry

theorem horner_rule_correctness (x : ℝ) :
  horner_v3 x = ((x + 3) * x - 1) * x := by sorry

theorem main_theorem : f 2 = ((((2 + 3) * 2 - 1) * 2 + 2) * 2 - 1) := by sorry

end NUMINAMATH_CALUDE_horner_rule_v3_equals_18_horner_rule_correctness_main_theorem_l1446_144627


namespace NUMINAMATH_CALUDE_prime_sum_equality_l1446_144607

theorem prime_sum_equality (n : ℕ) : 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p * (p + 1) + q * (q + 1) = n * (n + 1)) → 
  n = 3 ∨ n = 6 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_equality_l1446_144607


namespace NUMINAMATH_CALUDE_final_value_is_990_l1446_144668

def loop_calculation (s i : ℕ) : ℕ :=
  if i ≥ 9 then loop_calculation (s * i) (i - 1)
  else s

theorem final_value_is_990 : loop_calculation 1 11 = 990 := by
  sorry

end NUMINAMATH_CALUDE_final_value_is_990_l1446_144668


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1446_144699

theorem inequality_solution_set (t m : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + t < 0 ↔ 1 < x ∧ x < m) → 
  t = 2 ∧ m = 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1446_144699


namespace NUMINAMATH_CALUDE_shopkeeper_total_cards_l1446_144651

/-- The number of cards in a standard deck of playing cards -/
def standard_deck_size : ℕ := 52

/-- The number of complete decks the shopkeeper has -/
def complete_decks : ℕ := 6

/-- The number of additional cards the shopkeeper has -/
def additional_cards : ℕ := 7

/-- Theorem: The total number of cards the shopkeeper has is 319 -/
theorem shopkeeper_total_cards : 
  complete_decks * standard_deck_size + additional_cards = 319 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_total_cards_l1446_144651


namespace NUMINAMATH_CALUDE_soap_box_height_l1446_144633

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Theorem: Given the dimensions of a carton and soap boxes, and the maximum number of soap boxes
    that can fit in the carton, prove that the height of a soap box is 1 inch. -/
theorem soap_box_height
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (max_boxes : ℕ)
  (h_carton_length : carton.length = 30)
  (h_carton_width : carton.width = 42)
  (h_carton_height : carton.height = 60)
  (h_soap_length : soap.length = 7)
  (h_soap_width : soap.width = 6)
  (h_max_boxes : max_boxes = 360)
  : soap.height = 1 :=
by sorry

end NUMINAMATH_CALUDE_soap_box_height_l1446_144633


namespace NUMINAMATH_CALUDE_rihanna_shopping_theorem_l1446_144606

def calculate_remaining_money (initial_amount : ℕ) (mango_count : ℕ) (juice_count : ℕ) (mango_price : ℕ) (juice_price : ℕ) : ℕ :=
  initial_amount - (mango_count * mango_price + juice_count * juice_price)

theorem rihanna_shopping_theorem (initial_amount : ℕ) (mango_count : ℕ) (juice_count : ℕ) (mango_price : ℕ) (juice_price : ℕ) :
  calculate_remaining_money initial_amount mango_count juice_count mango_price juice_price =
  initial_amount - (mango_count * mango_price + juice_count * juice_price) :=
by
  sorry

#eval calculate_remaining_money 50 6 6 3 3

end NUMINAMATH_CALUDE_rihanna_shopping_theorem_l1446_144606


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1446_144680

theorem purely_imaginary_complex_number (a : ℝ) :
  (Complex.I * Complex.im (a * (1 + Complex.I) - 2) = a * (1 + Complex.I) - 2) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1446_144680


namespace NUMINAMATH_CALUDE_expression_value_l1446_144653

theorem expression_value : 
  let x : ℤ := 25
  let y : ℤ := 30
  let z : ℤ := 10
  (x - (y - z)) - ((x - y) - z) = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1446_144653


namespace NUMINAMATH_CALUDE_irrationality_of_pi_and_rationality_of_others_l1446_144684

-- Define irrational numbers
def IsIrrational (x : ℝ) : Prop :=
  ∀ a b : ℤ, b ≠ 0 → x ≠ a / b

-- State the theorem
theorem irrationality_of_pi_and_rationality_of_others :
  IsIrrational Real.pi ∧ ¬IsIrrational 0 ∧ ¬IsIrrational (-1/3) ∧ ¬IsIrrational (3/2) :=
sorry

end NUMINAMATH_CALUDE_irrationality_of_pi_and_rationality_of_others_l1446_144684


namespace NUMINAMATH_CALUDE_correct_factorization_l1446_144617

theorem correct_factorization (x y : ℝ) : x * (x - y) + y * (y - x) = (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1446_144617


namespace NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_one_l1446_144687

theorem sin_50_plus_sqrt3_tan_10_equals_one :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_one_l1446_144687


namespace NUMINAMATH_CALUDE_find_x_value_l1446_144663

theorem find_x_value (A B : Set ℝ) (x : ℝ) : 
  A = {-1, 1} → 
  B = {0, 1, x-1} → 
  A ⊆ B → 
  x = 0 := by
sorry

end NUMINAMATH_CALUDE_find_x_value_l1446_144663


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l1446_144683

theorem power_fraction_simplification : (2^2020 + 2^2018) / (2^2020 - 2^2018) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l1446_144683


namespace NUMINAMATH_CALUDE_not_right_triangle_4_6_11_l1446_144629

/-- Checks if three line segments can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem: The line segments 4, 6, and 11 cannot form a right triangle -/
theorem not_right_triangle_4_6_11 : ¬ is_right_triangle 4 6 11 := by
  sorry

#check not_right_triangle_4_6_11

end NUMINAMATH_CALUDE_not_right_triangle_4_6_11_l1446_144629


namespace NUMINAMATH_CALUDE_max_product_divisible_by_55_l1446_144679

/-- Represents a four-digit number in the form 11,0ab -/
structure Number11_0ab where
  a : Nat
  b : Nat
  a_single_digit : a < 10
  b_single_digit : b < 10

/-- Check if a number in the form 11,0ab is divisible by 55 -/
def isDivisibleBy55 (n : Number11_0ab) : Prop :=
  (11000 + 100 * n.a + n.b) % 55 = 0

/-- The maximum product of a and b for numbers divisible by 55 -/
def maxProduct : Nat :=
  25

theorem max_product_divisible_by_55 :
  ∀ n : Number11_0ab, isDivisibleBy55 n → n.a * n.b ≤ maxProduct :=
by sorry

end NUMINAMATH_CALUDE_max_product_divisible_by_55_l1446_144679


namespace NUMINAMATH_CALUDE_rhombus_height_l1446_144696

/-- A rhombus with diagonals of length 6 and 8 has a height of 24/5 -/
theorem rhombus_height (d₁ d₂ h : ℝ) (hd₁ : d₁ = 6) (hd₂ : d₂ = 8) :
  d₁ * d₂ = 4 * h * (d₁^2 / 4 + d₂^2 / 4).sqrt → h = 24 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_height_l1446_144696


namespace NUMINAMATH_CALUDE_product_repeating_third_and_nine_l1446_144664

/-- The repeating decimal 0.3̄ -/
def repeating_third : ℚ := 1 / 3

/-- The theorem stating that 0.3̄ * 9 = 3 -/
theorem product_repeating_third_and_nine : repeating_third * 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_repeating_third_and_nine_l1446_144664


namespace NUMINAMATH_CALUDE_segments_intersection_l1446_144692

-- Define the number of segments
def n : ℕ := 1977

-- Define the type for segments
def Segment : Type := ℕ → Set ℝ

-- Define the property of intersection
def intersects (s1 s2 : Set ℝ) : Prop := ∃ x, x ∈ s1 ∧ x ∈ s2

-- State the theorem
theorem segments_intersection 
  (A B : Segment) 
  (h1 : ∀ k ∈ Finset.range n, intersects (A k) (B ((k + n - 1) % n)))
  (h2 : ∀ k ∈ Finset.range n, intersects (A k) (B ((k + 1) % n)))
  (h3 : intersects (A (n - 1)) (B 0))
  (h4 : intersects (A 0) (B (n - 1)))
  : ∃ k ∈ Finset.range n, intersects (A k) (B k) :=
by sorry

end NUMINAMATH_CALUDE_segments_intersection_l1446_144692


namespace NUMINAMATH_CALUDE_uncle_fyodor_wins_l1446_144672

/-- Represents the state of a sandwich (with or without sausage) -/
inductive SandwichState
  | WithSausage
  | WithoutSausage

/-- Represents a player in the game -/
inductive Player
  | UncleFyodor
  | Matroskin

/-- The game state -/
structure GameState where
  sandwiches : List SandwichState
  currentPlayer : Player
  fyodorMoves : Nat
  matroskinMoves : Nat

/-- A move in the game -/
inductive Move
  | EatSandwich : Move  -- For Uncle Fyodor
  | RemoveSausage : Nat → Move  -- For Matroskin, with sandwich index

/-- Function to apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Function to check if the game is over -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Function to determine the winner -/
def getWinner (state : GameState) : Option Player :=
  sorry

/-- Theorem stating that Uncle Fyodor can always win for N = 2^100 - 1 -/
theorem uncle_fyodor_wins :
  ∀ (initialState : GameState),
    initialState.sandwiches.length = 100 * (2^100 - 1) →
    initialState.currentPlayer = Player.UncleFyodor →
    initialState.fyodorMoves = 0 →
    initialState.matroskinMoves = 0 →
    ∀ (matroskinStrategy : GameState → Move),
      ∃ (fyodorStrategy : GameState → Move),
        let finalState := sorry  -- Play out the game using the strategies
        getWinner finalState = some Player.UncleFyodor :=
  sorry


end NUMINAMATH_CALUDE_uncle_fyodor_wins_l1446_144672


namespace NUMINAMATH_CALUDE_equal_chord_circle_equation_l1446_144623

/-- A circle passing through two given points with equal chord lengths on coordinate axes -/
structure EqualChordCircle where
  -- Center of the circle
  center : ℝ × ℝ
  -- Radius of the circle
  radius : ℝ
  -- The circle passes through P(1, 2)
  passes_through_P : (center.1 - 1)^2 + (center.2 - 2)^2 = radius^2
  -- The circle passes through Q(-2, 3)
  passes_through_Q : (center.1 + 2)^2 + (center.2 - 3)^2 = radius^2
  -- Equal chord lengths on coordinate axes
  equal_chords : (center.1)^2 + (radius^2 - center.1^2) = (center.2)^2 + (radius^2 - center.2^2)

/-- The theorem stating that the circle has one of the two specific equations -/
theorem equal_chord_circle_equation (c : EqualChordCircle) :
  ((c.center.1 = -2 ∧ c.center.2 = -2 ∧ c.radius^2 = 25) ∨
   (c.center.1 = -1 ∧ c.center.2 = 1 ∧ c.radius^2 = 5)) :=
sorry

end NUMINAMATH_CALUDE_equal_chord_circle_equation_l1446_144623


namespace NUMINAMATH_CALUDE_cricket_bat_selling_price_l1446_144631

/-- Calculates the selling price of a cricket bat given the profit and profit percentage -/
theorem cricket_bat_selling_price (profit : ℝ) (profit_percentage : ℝ) :
  profit = 225 →
  profit_percentage = 36 →
  ∃ (cost_price selling_price : ℝ),
    cost_price = profit * 100 / profit_percentage ∧
    selling_price = cost_price + profit ∧
    selling_price = 850 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_selling_price_l1446_144631


namespace NUMINAMATH_CALUDE_tan_alpha_eq_one_third_l1446_144665

theorem tan_alpha_eq_one_third (α : Real) (h : Real.tan (α + π/4) = 2) : Real.tan α = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_eq_one_third_l1446_144665


namespace NUMINAMATH_CALUDE_limit_of_expression_l1446_144618

theorem limit_of_expression (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N →
    |((4 * (n : ℝ)^2 + 4 * n - 1) / (4 * (n : ℝ)^2 + 2 * n + 3))^(1 - 2 * n) - Real.exp (-1)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_expression_l1446_144618


namespace NUMINAMATH_CALUDE_consecutive_cubes_to_consecutive_squares_l1446_144662

theorem consecutive_cubes_to_consecutive_squares (A : ℕ) :
  (∃ k : ℕ, A^2 = (k + 1)^3 - k^3) →
  (∃ m : ℕ, A = m^2 + (m + 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_cubes_to_consecutive_squares_l1446_144662


namespace NUMINAMATH_CALUDE_symmetric_function_g_l1446_144657

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 1 - 2

-- Define the symmetry condition
def is_symmetric_about (g : ℝ → ℝ) (p : ℝ × ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y, g x = y ↔ f (2 * p.1 - x) = 2 * p.2 - y

-- Theorem statement
theorem symmetric_function_g : 
  ∃ g : ℝ → ℝ, is_symmetric_about g (1, 2) f ∧ (∀ x, g x = 3 * x - 1) :=
sorry

end NUMINAMATH_CALUDE_symmetric_function_g_l1446_144657


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l1446_144689

theorem upstream_downstream_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 63) 
  (h2 : stream_speed = 21) : 
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l1446_144689


namespace NUMINAMATH_CALUDE_tank_filling_time_l1446_144619

/-- Represents the time (in hours) it takes to fill the tank without the hole -/
def T : ℝ := 15

/-- Represents the time (in hours) it takes to fill the tank with the hole -/
def fill_time_with_hole : ℝ := 20

/-- Represents the time (in hours) it takes for the hole to empty the full tank -/
def empty_time : ℝ := 60

theorem tank_filling_time :
  (1 / T - 1 / empty_time = 1 / fill_time_with_hole) ∧
  (T > 0) ∧ (fill_time_with_hole > 0) ∧ (empty_time > 0) :=
sorry

end NUMINAMATH_CALUDE_tank_filling_time_l1446_144619


namespace NUMINAMATH_CALUDE_exists_fib_with_three_trailing_zeros_l1446_144608

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- State the theorem
theorem exists_fib_with_three_trailing_zeros :
  ∃ n : ℕ, fib n % 1000 = 0 ∧ fib (n + 1) % 1000 = 0 ∧ fib (n + 2) % 1000 = 0 := by
  sorry


end NUMINAMATH_CALUDE_exists_fib_with_three_trailing_zeros_l1446_144608


namespace NUMINAMATH_CALUDE_sum_of_cubes_theorem_l1446_144635

theorem sum_of_cubes_theorem (a b : ℤ) : 
  a * b = 12 → a^3 + b^3 = 91 → a^3 + b^3 = 91 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_theorem_l1446_144635


namespace NUMINAMATH_CALUDE_rectangular_region_area_l1446_144682

/-- The area of a rectangular region enclosed by lines derived from given equations -/
theorem rectangular_region_area (a : ℝ) (ha : a > 0) :
  let eq1 (x y : ℝ) := (2 * x - a * y)^2 = 25 * a^2
  let eq2 (x y : ℝ) := (5 * a * x + 2 * y)^2 = 36 * a^2
  let area := (120 * a^2) / Real.sqrt (100 * a^2 + 16 + 100 * a^4)
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    eq1 x1 y1 ∧ eq1 x2 y2 ∧ eq1 x3 y3 ∧ eq1 x4 y4 ∧
    eq2 x1 y1 ∧ eq2 x2 y2 ∧ eq2 x3 y3 ∧ eq2 x4 y4 ∧
    (x1 - x2) * (y1 - y3) = area :=
by sorry

end NUMINAMATH_CALUDE_rectangular_region_area_l1446_144682


namespace NUMINAMATH_CALUDE_popcorn_probability_l1446_144611

theorem popcorn_probability (white_ratio : ℚ) (yellow_ratio : ℚ) 
  (white_pop_prob : ℚ) (yellow_pop_prob : ℚ) :
  white_ratio = 3/4 →
  yellow_ratio = 1/4 →
  white_pop_prob = 1/3 →
  yellow_pop_prob = 3/4 →
  let white_and_pop := white_ratio * white_pop_prob
  let yellow_and_pop := yellow_ratio * yellow_pop_prob
  let total_pop := white_and_pop + yellow_and_pop
  (white_and_pop / total_pop) = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_probability_l1446_144611


namespace NUMINAMATH_CALUDE_diploma_percentage_theorem_l1446_144620

/-- Represents the four income groups in country Z -/
inductive IncomeGroup
  | Low
  | LowerMiddle
  | UpperMiddle
  | High

/-- Returns the population percentage for a given income group -/
def population_percentage (group : IncomeGroup) : Real :=
  match group with
  | IncomeGroup.Low => 0.25
  | IncomeGroup.LowerMiddle => 0.35
  | IncomeGroup.UpperMiddle => 0.25
  | IncomeGroup.High => 0.15

/-- Returns the percentage of people with a university diploma for a given income group -/
def diploma_percentage (group : IncomeGroup) : Real :=
  match group with
  | IncomeGroup.Low => 0.05
  | IncomeGroup.LowerMiddle => 0.35
  | IncomeGroup.UpperMiddle => 0.60
  | IncomeGroup.High => 0.80

/-- Calculates the total percentage of the population with a university diploma -/
def total_diploma_percentage : Real :=
  (population_percentage IncomeGroup.Low * diploma_percentage IncomeGroup.Low) +
  (population_percentage IncomeGroup.LowerMiddle * diploma_percentage IncomeGroup.LowerMiddle) +
  (population_percentage IncomeGroup.UpperMiddle * diploma_percentage IncomeGroup.UpperMiddle) +
  (population_percentage IncomeGroup.High * diploma_percentage IncomeGroup.High)

theorem diploma_percentage_theorem :
  total_diploma_percentage = 0.405 := by
  sorry

end NUMINAMATH_CALUDE_diploma_percentage_theorem_l1446_144620


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_two_dividing_32_factorial_l1446_144644

def largest_power_of_two_dividing_factorial (n : ℕ) : ℕ :=
  (n / 2) + (n / 4) + (n / 8) + (n / 16) + (n / 32)

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_largest_power_of_two_dividing_32_factorial :
  ones_digit (2^(largest_power_of_two_dividing_factorial 32)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_two_dividing_32_factorial_l1446_144644


namespace NUMINAMATH_CALUDE_midpoint_on_grid_l1446_144638

theorem midpoint_on_grid (points : Fin 5 → ℤ × ℤ) :
  ∃ i j, i ≠ j ∧ i < 5 ∧ j < 5 ∧
  (((points i).1 + (points j).1) % 2 = 0) ∧
  (((points i).2 + (points j).2) % 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_midpoint_on_grid_l1446_144638


namespace NUMINAMATH_CALUDE_smallest_x_value_l1446_144674

theorem smallest_x_value (x : ℝ) : 
  (4 * x^2 + 6 * x + 1 = 5) → x ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l1446_144674


namespace NUMINAMATH_CALUDE_distance_between_points_l1446_144605

theorem distance_between_points : 
  let pointA : ℝ × ℝ := (1, 2)
  let pointB : ℝ × ℝ := (5, 7)
  Real.sqrt ((pointB.1 - pointA.1)^2 + (pointB.2 - pointA.2)^2) = Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1446_144605


namespace NUMINAMATH_CALUDE_wild_ducks_geese_meeting_l1446_144661

/-- The number of days it takes wild ducks to fly from South Sea to North Sea -/
def wild_ducks_days : ℕ := 7

/-- The number of days it takes geese to fly from North Sea to South Sea -/
def geese_days : ℕ := 9

/-- The equation representing the meeting of wild ducks and geese -/
def meeting_equation (x : ℝ) : Prop :=
  (1 / wild_ducks_days : ℝ) * x + (1 / geese_days : ℝ) * x = 1

/-- Theorem stating that the solution to the meeting equation represents
    the number of days it takes for wild ducks and geese to meet -/
theorem wild_ducks_geese_meeting :
  ∃ x : ℝ, x > 0 ∧ meeting_equation x ∧
    ∀ y : ℝ, y > 0 ∧ meeting_equation y → x = y :=
sorry

end NUMINAMATH_CALUDE_wild_ducks_geese_meeting_l1446_144661


namespace NUMINAMATH_CALUDE_icosagon_diagonals_l1446_144615

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular icosagon has 20 sides -/
def icosagon_sides : ℕ := 20

theorem icosagon_diagonals :
  num_diagonals icosagon_sides = 170 := by
  sorry

end NUMINAMATH_CALUDE_icosagon_diagonals_l1446_144615


namespace NUMINAMATH_CALUDE_square_diagonal_l1446_144616

theorem square_diagonal (A : ℝ) (h : A = 800) :
  ∃ d : ℝ, d = 40 ∧ d^2 = 2 * A :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_l1446_144616


namespace NUMINAMATH_CALUDE_total_homework_time_l1446_144647

def jacob_time : ℕ := 18

def greg_time (jacob_time : ℕ) : ℕ := jacob_time - 6

def patrick_time (greg_time : ℕ) : ℕ := 2 * greg_time - 4

def samantha_time (patrick_time : ℕ) : ℕ := (3 * patrick_time) / 2

theorem total_homework_time :
  jacob_time + greg_time jacob_time + patrick_time (greg_time jacob_time) + samantha_time (patrick_time (greg_time jacob_time)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_homework_time_l1446_144647


namespace NUMINAMATH_CALUDE_parabola_square_min_area_l1446_144675

/-- A square in a Cartesian plane with vertices on two parabolas -/
structure ParabolaSquare where
  /-- x-coordinate of a vertex on y = x^2 -/
  a : ℝ
  /-- The square's side length -/
  s : ℝ
  /-- Two opposite vertices lie on y = x^2 -/
  h1 : (a, a^2) ∈ {p : ℝ × ℝ | p.2 = p.1^2}
  h2 : (-a, a^2) ∈ {p : ℝ × ℝ | p.2 = p.1^2}
  /-- The other two opposite vertices lie on y = -x^2 + 4 -/
  h3 : (a, -a^2 + 4) ∈ {p : ℝ × ℝ | p.2 = -p.1^2 + 4}
  h4 : (-a, -a^2 + 4) ∈ {p : ℝ × ℝ | p.2 = -p.1^2 + 4}
  /-- The side length is the distance between vertices -/
  h5 : s^2 = (2*a)^2 + (2*a^2 - 4)^2

/-- The smallest possible area of the ParabolaSquare is 4 -/
theorem parabola_square_min_area :
  ∀ (ps : ParabolaSquare), ∃ (min_ps : ParabolaSquare), min_ps.s^2 = 4 ∧ ∀ (ps' : ParabolaSquare), ps'.s^2 ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_square_min_area_l1446_144675


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l1446_144610

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def has_six_consecutive_nonprimes (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ 
    ∀ i : ℕ, i ≥ k ∧ i < k + 6 → ¬(is_prime i)

theorem smallest_prime_after_six_nonprimes :
  (is_prime 97) ∧ 
  (has_six_consecutive_nonprimes 96) ∧ 
  (∀ p : ℕ, p < 97 → ¬(is_prime p ∧ has_six_consecutive_nonprimes (p - 1))) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l1446_144610


namespace NUMINAMATH_CALUDE_no_solution_implies_a_less_than_one_l1446_144622

theorem no_solution_implies_a_less_than_one (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 1| + x ≤ a)) → a < 1 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_less_than_one_l1446_144622


namespace NUMINAMATH_CALUDE_vertical_shift_theorem_l1446_144630

theorem vertical_shift_theorem (f : ℝ → ℝ) :
  ∀ x y : ℝ, y = f x + 3 ↔ ∃ y₀ : ℝ, y₀ = f x ∧ y = y₀ + 3 := by sorry

end NUMINAMATH_CALUDE_vertical_shift_theorem_l1446_144630


namespace NUMINAMATH_CALUDE_weekly_earnings_increase_l1446_144640

/-- Calculates the percentage increase between two amounts -/
def percentageIncrease (originalAmount newAmount : ℚ) : ℚ :=
  ((newAmount - originalAmount) / originalAmount) * 100

theorem weekly_earnings_increase (originalAmount newAmount : ℚ) 
  (h1 : originalAmount = 40)
  (h2 : newAmount = 80) :
  percentageIncrease originalAmount newAmount = 100 := by
  sorry

#eval percentageIncrease 40 80

end NUMINAMATH_CALUDE_weekly_earnings_increase_l1446_144640


namespace NUMINAMATH_CALUDE_chloe_first_round_points_l1446_144655

/-- Represents the points scored in a trivia game. -/
structure TriviaProblem where
  first_round : ℤ
  second_round : ℤ
  last_round : ℤ
  total_points : ℤ

/-- The solution to Chloe's trivia game problem. -/
theorem chloe_first_round_points (game : TriviaProblem) 
  (h1 : game.second_round = 50)
  (h2 : game.last_round = -4)
  (h3 : game.total_points = 86)
  (h4 : game.first_round + game.second_round + game.last_round = game.total_points) :
  game.first_round = 40 := by
  sorry

end NUMINAMATH_CALUDE_chloe_first_round_points_l1446_144655


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1446_144601

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a₃ + a₅ = 16, then a₄ = 8 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 5 = 16) : 
  a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1446_144601


namespace NUMINAMATH_CALUDE_total_people_present_l1446_144613

/-- Represents the number of associate professors -/
def associate_profs : ℕ := sorry

/-- Represents the number of assistant professors -/
def assistant_profs : ℕ := sorry

/-- Total number of pencils brought to the meeting -/
def total_pencils : ℕ := 10

/-- Total number of charts brought to the meeting -/
def total_charts : ℕ := 14

/-- Theorem stating the total number of people present at the meeting -/
theorem total_people_present : associate_profs + assistant_profs = 8 :=
  sorry

end NUMINAMATH_CALUDE_total_people_present_l1446_144613


namespace NUMINAMATH_CALUDE_inflection_points_collinear_l1446_144621

/-- The function f(x) = 9x^5 - 30x^3 + 19x -/
def f (x : ℝ) : ℝ := 9*x^5 - 30*x^3 + 19*x

/-- The inflection points of f(x) -/
def inflection_points : List (ℝ × ℝ) := [(-1, 2), (0, 0), (1, -2)]

/-- Theorem: The inflection points of f(x) are collinear -/
theorem inflection_points_collinear : 
  let points := inflection_points
  ∃ (m c : ℝ), ∀ (x y : ℝ), (x, y) ∈ points → y = m * x + c :=
by sorry

end NUMINAMATH_CALUDE_inflection_points_collinear_l1446_144621


namespace NUMINAMATH_CALUDE_least_months_to_triple_l1446_144648

/-- The initial borrowed amount in dollars -/
def initial_amount : ℝ := 1500

/-- The monthly interest rate as a decimal -/
def interest_rate : ℝ := 0.06

/-- The factor by which the borrowed amount increases each month -/
def growth_factor : ℝ := 1 + interest_rate

/-- The amount owed after t months -/
def amount_owed (t : ℕ) : ℝ := initial_amount * growth_factor ^ t

/-- Predicate that checks if the amount owed exceeds three times the initial amount -/
def exceeds_triple (t : ℕ) : Prop := amount_owed t > 3 * initial_amount

theorem least_months_to_triple :
  (∀ m : ℕ, m < 20 → ¬(exceeds_triple m)) ∧ exceeds_triple 20 :=
sorry

end NUMINAMATH_CALUDE_least_months_to_triple_l1446_144648


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1446_144643

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1446_144643


namespace NUMINAMATH_CALUDE_binomial_20_choose_6_l1446_144632

theorem binomial_20_choose_6 : Nat.choose 20 6 = 38760 := by sorry

end NUMINAMATH_CALUDE_binomial_20_choose_6_l1446_144632


namespace NUMINAMATH_CALUDE_discount_percentage_l1446_144614

theorem discount_percentage (cost_price : ℝ) (profit_with_discount : ℝ) (profit_without_discount : ℝ)
  (h1 : profit_with_discount = 42.5)
  (h2 : profit_without_discount = 50)
  : (((1 + profit_without_discount / 100) * cost_price - (1 + profit_with_discount / 100) * cost_price) /
     ((1 + profit_without_discount / 100) * cost_price)) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l1446_144614


namespace NUMINAMATH_CALUDE_max_value_cos_sin_linear_combination_l1446_144693

theorem max_value_cos_sin_linear_combination (a b φ : ℝ) :
  (∀ θ : ℝ, a * Real.cos (θ - φ) + b * Real.sin (θ - φ) ≤ Real.sqrt (a^2 + b^2)) ∧
  (∃ θ : ℝ, a * Real.cos (θ - φ) + b * Real.sin (θ - φ) = Real.sqrt (a^2 + b^2)) := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_linear_combination_l1446_144693


namespace NUMINAMATH_CALUDE_harmonic_mean_closest_integer_l1446_144678

theorem harmonic_mean_closest_integer :
  ∃ (h : ℝ), 
    (h = 2 / ((1 : ℝ)⁻¹ + (2023 : ℝ)⁻¹)) ∧ 
    (∀ n : ℤ, n ≠ 2 → |h - 2| < |h - (n : ℝ)|) := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_closest_integer_l1446_144678


namespace NUMINAMATH_CALUDE_janessas_cards_l1446_144626

/-- The number of cards Janessa's father gave her. -/
def fathers_cards : ℕ := by sorry

theorem janessas_cards :
  let initial_cards : ℕ := 4
  let ebay_cards : ℕ := 36
  let discarded_cards : ℕ := 4
  let cards_to_dexter : ℕ := 29
  let cards_kept : ℕ := 20
  fathers_cards = 13 := by sorry

end NUMINAMATH_CALUDE_janessas_cards_l1446_144626


namespace NUMINAMATH_CALUDE_parabola_equation_l1446_144603

-- Define the parabola
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => x^2 = 2 * p * y

-- Define points on the parabola
def Point := ℝ × ℝ

-- Define the problem setup
structure ParabolaProblem where
  parabola : Parabola
  F : Point
  A : Point
  B : Point
  C : Point
  D : Point
  l : Point → Prop

-- Define the conditions
def satisfies_conditions (prob : ParabolaProblem) : Prop :=
  let (xf, yf) := prob.F
  let (xa, ya) := prob.A
  let (xb, yb) := prob.B
  let (xc, yc) := prob.C
  let (xd, yd) := prob.D
  xf = 0 ∧ yf > 0 ∧
  prob.parabola.eq xa ya ∧
  prob.parabola.eq xb yb ∧
  prob.l prob.A ∧ prob.l prob.B ∧
  xc = xa ∧ xd = xb ∧
  (ya - yf)^2 + xa^2 = 4 * ((yf - yb)^2 + xb^2) ∧
  (xd - xc) * (xa - xb) + (yd - yc) * (ya - yb) = 72

-- Theorem statement
theorem parabola_equation (prob : ParabolaProblem) 
  (h : satisfies_conditions prob) : 
  prob.parabola.p = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1446_144603


namespace NUMINAMATH_CALUDE_fifth_patient_cure_rate_l1446_144602

theorem fifth_patient_cure_rate 
  (cure_rate : ℝ) 
  (h_cure_rate : cure_rate = 1/5) 
  (first_four_patients : Fin 4 → Bool) 
  : ℝ :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_patient_cure_rate_l1446_144602


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1446_144667

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x₀ : ℝ, x₀ - 2 > Real.log x₀) ↔ (∀ x : ℝ, x - 2 ≤ Real.log x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1446_144667


namespace NUMINAMATH_CALUDE_total_books_count_l1446_144660

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 6

/-- The number of shelves with mystery books -/
def mystery_shelves : ℕ := 5

/-- The number of shelves with picture books -/
def picture_shelves : ℕ := 4

/-- The total number of books -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem total_books_count : total_books = 54 := by sorry

end NUMINAMATH_CALUDE_total_books_count_l1446_144660


namespace NUMINAMATH_CALUDE_horner_rule_v2_l1446_144650

def horner_polynomial (x : ℚ) : ℚ := 1 + 2*x + x^2 - 3*x^3 + 2*x^4

def horner_v2 (x : ℚ) : ℚ :=
  let v1 := 2*x^3 - 3*x^2 + x
  v1 * x + 2

theorem horner_rule_v2 :
  horner_v2 (-1) = -4 :=
by sorry

end NUMINAMATH_CALUDE_horner_rule_v2_l1446_144650


namespace NUMINAMATH_CALUDE_problem_solution_l1446_144641

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -12)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 8) :
  b / (a + b) + c / (b + c) + a / (c + a) = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1446_144641


namespace NUMINAMATH_CALUDE_complex_number_coordinate_l1446_144694

theorem complex_number_coordinate : 
  let z : ℂ := 1 + (1 / Complex.I)
  (z.re = 1 ∧ z.im = -1) := by sorry

end NUMINAMATH_CALUDE_complex_number_coordinate_l1446_144694


namespace NUMINAMATH_CALUDE_simultaneous_integers_l1446_144636

theorem simultaneous_integers (t : ℤ) : 
  let x : ℤ := 60 * t + 1
  (∃ (k₁ k₂ k₃ : ℤ), (2 * x + 1) / 3 = k₁ ∧ (3 * x + 1) / 4 = k₂ ∧ (4 * x + 1) / 5 = k₃) ∧
  (∀ (y : ℤ), y ≠ x → ¬(∃ (k₁ k₂ k₃ : ℤ), (2 * y + 1) / 3 = k₁ ∧ (3 * y + 1) / 4 = k₂ ∧ (4 * y + 1) / 5 = k₃)) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_integers_l1446_144636


namespace NUMINAMATH_CALUDE_jade_ball_problem_l1446_144609

/-- Represents the state of boxes as a list of natural numbers (0-6) -/
def BoxState := List Nat

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : Nat) : BoxState :=
  sorry

/-- Counts the number of carries (resets) needed to increment from 1 to n in base 7 -/
def countCarries (n : Nat) : Nat :=
  sorry

/-- Sums the digits in a BoxState -/
def sumDigits (state : BoxState) : Nat :=
  sorry

theorem jade_ball_problem (n : Nat) : 
  n = 1876 → 
  sumDigits (toBase7 n) = 10 ∧ 
  countCarries n = 3 := by
  sorry

end NUMINAMATH_CALUDE_jade_ball_problem_l1446_144609


namespace NUMINAMATH_CALUDE_convoy_problem_l1446_144637

/-- Represents the convoy of vehicles -/
structure Convoy where
  num_vehicles : ℕ
  departure_interval : ℚ
  first_departure : ℚ
  stop_time : ℚ
  speed : ℚ

/-- Calculate the travel time of the last vehicle in the convoy -/
def last_vehicle_travel_time (c : Convoy) : ℚ :=
  c.stop_time - (c.first_departure + (c.num_vehicles - 1) * c.departure_interval)

/-- Calculate the total distance traveled by the convoy -/
def total_distance_traveled (c : Convoy) : ℚ :=
  let total_time := c.num_vehicles * (c.stop_time - c.first_departure) - 
    (c.num_vehicles * (c.num_vehicles - 1) / 2) * c.departure_interval
  total_time * c.speed

/-- The main theorem statement -/
theorem convoy_problem (c : Convoy) 
  (h1 : c.num_vehicles = 15)
  (h2 : c.departure_interval = 1/6)
  (h3 : c.first_departure = 2)
  (h4 : c.stop_time = 6)
  (h5 : c.speed = 60) : 
  last_vehicle_travel_time c = 5/3 ∧ 
  total_distance_traveled c = 2550 := by
  sorry

#eval last_vehicle_travel_time ⟨15, 1/6, 2, 6, 60⟩
#eval total_distance_traveled ⟨15, 1/6, 2, 6, 60⟩

end NUMINAMATH_CALUDE_convoy_problem_l1446_144637


namespace NUMINAMATH_CALUDE_binomial_coefficient_inequality_l1446_144656

theorem binomial_coefficient_inequality (n k h : ℕ) (h1 : n ≥ k + h) :
  Nat.choose n (k + h) ≥ Nat.choose (n - k) h :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_inequality_l1446_144656


namespace NUMINAMATH_CALUDE_sound_speed_model_fits_data_sound_speed_model_unique_l1446_144671

/-- Represents the relationship between temperature and sound speed -/
def sound_speed_model (x : ℝ) : ℝ := 330 + 0.6 * x

/-- The set of data points for temperature and sound speed -/
def data_points : List (ℝ × ℝ) := [
  (-20, 318), (-10, 324), (0, 330), (10, 336), (20, 342), (30, 348)
]

/-- Theorem stating that the sound_speed_model fits the given data points -/
theorem sound_speed_model_fits_data : 
  ∀ (point : ℝ × ℝ), point ∈ data_points → 
    sound_speed_model point.1 = point.2 := by
  sorry

/-- Theorem stating that the sound_speed_model is the unique linear model fitting the data -/
theorem sound_speed_model_unique : 
  ∀ (a b : ℝ), (∀ (point : ℝ × ℝ), point ∈ data_points → 
    a + b * point.1 = point.2) → a = 330 ∧ b = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_sound_speed_model_fits_data_sound_speed_model_unique_l1446_144671


namespace NUMINAMATH_CALUDE_similar_triangles_height_l1446_144625

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 9 →
  let scale_factor := Real.sqrt area_ratio
  let h_large := h_small * scale_factor
  h_small = 5 →
  h_large = 15 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l1446_144625


namespace NUMINAMATH_CALUDE_point_below_line_l1446_144642

theorem point_below_line (m : ℝ) : 
  ((-2 : ℝ) + m * (-1 : ℝ) - 1 < 0) ↔ (m < -3 ∨ m > 0) :=
by sorry

end NUMINAMATH_CALUDE_point_below_line_l1446_144642


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l1446_144645

theorem cube_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (z^3 + y^3) / (x^2 + x*y + y^2) + (x^3 + z^3) / (y^2 + y*z + z^2) + (y^3 + x^3) / (z^2 + z*x + x^2) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l1446_144645


namespace NUMINAMATH_CALUDE_prob_event_A_is_three_eighths_l1446_144676

/-- Represents the faces of the tetrahedron -/
inductive Face : Type
  | zero : Face
  | one : Face
  | two : Face
  | three : Face

/-- Converts a Face to its numerical value -/
def faceValue : Face → ℕ
  | Face.zero => 0
  | Face.one => 1
  | Face.two => 2
  | Face.three => 3

/-- Defines the event A: m^2 + n^2 ≤ 4 -/
def eventA (m n : Face) : Prop :=
  (faceValue m)^2 + (faceValue n)^2 ≤ 4

/-- The probability of event A occurring -/
def probEventA : ℚ := 3/8

/-- Theorem stating that the probability of event A is 3/8 -/
theorem prob_event_A_is_three_eighths :
  probEventA = 3/8 := by sorry

end NUMINAMATH_CALUDE_prob_event_A_is_three_eighths_l1446_144676


namespace NUMINAMATH_CALUDE_intersecting_planes_not_imply_intersecting_lines_l1446_144688

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perpendicular : Line → Plane → Prop)

-- Define the intersection relation for lines and for planes
variable (lines_intersect : Line → Line → Prop)
variable (planes_intersect : Plane → Plane → Prop)

-- State the theorem
theorem intersecting_planes_not_imply_intersecting_lines 
  (a b : Line) (α β : Plane) 
  (h1 : a ≠ b) 
  (h2 : α ≠ β) 
  (h3 : perpendicular a α) 
  (h4 : perpendicular b β) :
  ∃ (α β : Plane), planes_intersect α β ∧ ¬ lines_intersect a b :=
sorry

end NUMINAMATH_CALUDE_intersecting_planes_not_imply_intersecting_lines_l1446_144688


namespace NUMINAMATH_CALUDE_line_equation_l1446_144658

/-- A line passing through a point and intersecting axes -/
structure Line where
  -- Point that the line passes through
  P : ℝ × ℝ
  -- x-coordinate of intersection with positive x-axis
  C : ℝ
  -- y-coordinate of intersection with negative y-axis
  D : ℝ
  -- Condition that P lies on the line
  point_on_line : (P.1 / C) + (P.2 / (-D)) = 1
  -- Condition for positive x-axis intersection
  pos_x_axis : C > 0
  -- Condition for negative y-axis intersection
  neg_y_axis : D > 0
  -- Area condition
  area_condition : (1/2) * C * D = 2

/-- Theorem stating the equation of the line -/
theorem line_equation (l : Line) (h : l.P = (1, -1)) :
  ∃ (a b : ℝ), a * l.P.1 + b * l.P.2 + 2 = 0 ∧ 
               ∀ (x y : ℝ), a * x + b * y + 2 = 0 ↔ (x / l.C) + (y / (-l.D)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l1446_144658


namespace NUMINAMATH_CALUDE_dice_sum_possibilities_l1446_144628

/-- The number of dice being rolled -/
def num_dice : ℕ := 4

/-- The minimum value on a die face -/
def min_face : ℕ := 1

/-- The maximum value on a die face -/
def max_face : ℕ := 6

/-- The minimum possible sum when rolling the dice -/
def min_sum : ℕ := num_dice * min_face

/-- The maximum possible sum when rolling the dice -/
def max_sum : ℕ := num_dice * max_face

/-- The number of distinct possible sums when rolling the dice -/
def num_distinct_sums : ℕ := max_sum - min_sum + 1

theorem dice_sum_possibilities : num_distinct_sums = 21 := by
  sorry

end NUMINAMATH_CALUDE_dice_sum_possibilities_l1446_144628


namespace NUMINAMATH_CALUDE_x_4_sufficient_not_necessary_l1446_144691

def vector_a (x : ℝ) : Fin 2 → ℝ := ![x, 3]

theorem x_4_sufficient_not_necessary :
  (∀ x : ℝ, x = 4 → ‖vector_a x‖ = 5) ∧
  (∃ y : ℝ, y ≠ 4 ∧ ‖vector_a y‖ = 5) :=
by sorry

end NUMINAMATH_CALUDE_x_4_sufficient_not_necessary_l1446_144691


namespace NUMINAMATH_CALUDE_total_rainfall_2011_2012_l1446_144697

/-- Represents the average monthly rainfall in millimeters for a given year. -/
def AverageMonthlyRainfall : ℕ → ℝ
  | 2010 => 50.0
  | 2011 => AverageMonthlyRainfall 2010 + 3
  | 2012 => AverageMonthlyRainfall 2011 + 4
  | _ => 0  -- Default case for other years

/-- Calculates the total yearly rainfall given the average monthly rainfall. -/
def YearlyRainfall (year : ℕ) : ℝ :=
  AverageMonthlyRainfall year * 12

/-- Theorem stating the total rainfall in Clouddale for 2011 and 2012. -/
theorem total_rainfall_2011_2012 :
  YearlyRainfall 2011 + YearlyRainfall 2012 = 1320.0 := by
  sorry

#eval YearlyRainfall 2011 + YearlyRainfall 2012

end NUMINAMATH_CALUDE_total_rainfall_2011_2012_l1446_144697


namespace NUMINAMATH_CALUDE_cannot_form_triangle_l1446_144659

/-- A function that checks if three line segments can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: Three line segments with lengths 1, 2, and 3 cannot form a triangle --/
theorem cannot_form_triangle : ¬(can_form_triangle 1 2 3) := by
  sorry

end NUMINAMATH_CALUDE_cannot_form_triangle_l1446_144659


namespace NUMINAMATH_CALUDE_sum_powers_of_i_l1446_144639

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_powers_of_i :
  i^300 + i^301 + i^302 + i^303 + i^304 + i^305 + i^306 + i^307 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_powers_of_i_l1446_144639


namespace NUMINAMATH_CALUDE_sheep_buying_problem_l1446_144698

/-- Represents the sheep buying problem from "The Nine Chapters on the Mathematical Art" --/
theorem sheep_buying_problem (x y : ℤ) : 
  (∀ (contribution shortage : ℤ), contribution = 5 ∧ shortage = 45 → contribution * x + shortage = y) ∧
  (∀ (contribution surplus : ℤ), contribution = 7 ∧ surplus = 3 → contribution * x - surplus = y) ↔
  (5 * x + 45 = y ∧ 7 * x - 3 = y) :=
sorry


end NUMINAMATH_CALUDE_sheep_buying_problem_l1446_144698


namespace NUMINAMATH_CALUDE_ellipse_origin_inside_l1446_144634

theorem ellipse_origin_inside (k : ℝ) : 
  (∀ x y : ℝ, k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 = 0 → x^2 + y^2 > 0) →
  (k^2 * 0^2 + 0^2 - 4*k*0 + 2*k*0 + k^2 - 1 < 0) →
  0 < |k| ∧ |k| < 1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_origin_inside_l1446_144634


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1446_144673

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 6*I
  let z₂ : ℂ := 4 - 6*I
  (z₁ / z₂) + (z₂ / z₁) = -10/13 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1446_144673


namespace NUMINAMATH_CALUDE_family_d_savings_l1446_144600

/-- Calculates the percentage of money saved by a family given the number of passengers,
    planned spending, and cost per orange. -/
def percentage_saved (passengers : ℕ) (planned_spending : ℚ) (cost_per_orange : ℚ) : ℚ :=
  (passengers * cost_per_orange) / planned_spending * 100

/-- Theorem stating that Family D saves 55% of their planned spending. -/
theorem family_d_savings :
  percentage_saved 3 12 (22/10) = 55 := by
  sorry

end NUMINAMATH_CALUDE_family_d_savings_l1446_144600


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1446_144686

theorem fraction_evaluation (a b c : ℝ) (h : a^3 - b^3 + c^3 ≠ 0) :
  (a^6 - b^6 + c^6) / (a^3 - b^3 + c^3) = a^3 + b^3 + c^3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1446_144686


namespace NUMINAMATH_CALUDE_odd_prime_sum_divisors_count_l1446_144685

/-- Sum of positive integer divisors of n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Predicate for odd prime numbers -/
def is_odd_prime (n : ℕ) : Prop := sorry

/-- Count of numbers with odd prime sum of divisors -/
def count_odd_prime_sum_divisors : ℕ := sorry

theorem odd_prime_sum_divisors_count :
  count_odd_prime_sum_divisors = 5 := by sorry

end NUMINAMATH_CALUDE_odd_prime_sum_divisors_count_l1446_144685


namespace NUMINAMATH_CALUDE_regression_slope_effect_l1446_144670

/-- Represents a linear regression model --/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Predicted y value for a given x --/
def LinearRegression.predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.intercept + model.slope * x

theorem regression_slope_effect (model : LinearRegression) 
  (h : model.slope = -1 ∧ model.intercept = 2) :
  ∀ x : ℝ, model.predict (x + 1) = model.predict x - 1 := by
  sorry

#check regression_slope_effect

end NUMINAMATH_CALUDE_regression_slope_effect_l1446_144670


namespace NUMINAMATH_CALUDE_gcf_of_48_180_120_l1446_144677

theorem gcf_of_48_180_120 : Nat.gcd 48 (Nat.gcd 180 120) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_48_180_120_l1446_144677


namespace NUMINAMATH_CALUDE_trees_survived_difference_l1446_144690

theorem trees_survived_difference (initial_trees died_trees : ℕ) 
  (h1 : initial_trees = 11)
  (h2 : died_trees = 2) :
  initial_trees - died_trees - died_trees = 7 := by
  sorry

end NUMINAMATH_CALUDE_trees_survived_difference_l1446_144690


namespace NUMINAMATH_CALUDE_cloves_discrepancy_l1446_144654

/-- Represents the number of creatures that can be repelled by 3 cloves of garlic -/
structure RepelRatio :=
  (vampires : ℚ)
  (wights : ℚ)
  (vampire_bats : ℚ)

/-- Represents the number of creatures to be repelled -/
structure CreaturesToRepel :=
  (vampires : ℕ)
  (wights : ℕ)
  (vampire_bats : ℕ)

/-- Calculates the number of cloves needed based on the repel ratio and creatures to repel -/
def cloves_needed (ratio : RepelRatio) (creatures : CreaturesToRepel) : ℚ :=
  3 * (creatures.vampires / ratio.vampires + 
       creatures.wights / ratio.wights + 
       creatures.vampire_bats / ratio.vampire_bats)

/-- The main theorem stating that the calculated cloves needed is not equal to 72 -/
theorem cloves_discrepancy (ratio : RepelRatio) (creatures : CreaturesToRepel) :
  ratio.vampires = 1 →
  ratio.wights = 3 →
  ratio.vampire_bats = 8 →
  creatures.vampires = 30 →
  creatures.wights = 12 →
  creatures.vampire_bats = 40 →
  cloves_needed ratio creatures ≠ 72 := by
  sorry


end NUMINAMATH_CALUDE_cloves_discrepancy_l1446_144654


namespace NUMINAMATH_CALUDE_fraction_less_than_two_l1446_144646

theorem fraction_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_two_l1446_144646


namespace NUMINAMATH_CALUDE_sqrt_two_times_two_minus_sqrt_two_sqrt_six_div_sqrt_three_times_sqrt_twentyfour_sum_of_square_roots_squared_difference_minus_product_l1446_144695

-- Problem 1
theorem sqrt_two_times_two_minus_sqrt_two :
  Real.sqrt 2 * (2 - Real.sqrt 2) = 2 * Real.sqrt 2 - 2 := by sorry

-- Problem 2
theorem sqrt_six_div_sqrt_three_times_sqrt_twentyfour :
  Real.sqrt 6 / Real.sqrt 3 * Real.sqrt 24 = 4 * Real.sqrt 3 := by sorry

-- Problem 3
theorem sum_of_square_roots :
  Real.sqrt 54 + Real.sqrt 24 - Real.sqrt 18 + 2 * Real.sqrt (1/2) = 5 * Real.sqrt 6 - 2 * Real.sqrt 2 := by sorry

-- Problem 4
theorem squared_difference_minus_product :
  (Real.sqrt 2 - 1)^2 - (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) = 2 - 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_two_times_two_minus_sqrt_two_sqrt_six_div_sqrt_three_times_sqrt_twentyfour_sum_of_square_roots_squared_difference_minus_product_l1446_144695
