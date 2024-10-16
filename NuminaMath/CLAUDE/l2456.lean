import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2456_245624

/-- 
An arithmetic sequence is defined by its first term, common difference, and last term.
This theorem proves that an arithmetic sequence with first term -6, last term 38,
and common difference 4 has exactly 12 terms.
-/
theorem arithmetic_sequence_length 
  (a : ℤ) (d : ℤ) (l : ℤ) (n : ℕ) 
  (h1 : a = -6) 
  (h2 : d = 4) 
  (h3 : l = 38) 
  (h4 : l = a + (n - 1) * d) : n = 12 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2456_245624


namespace NUMINAMATH_CALUDE_chord_intersection_sum_of_squares_l2456_245632

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit
def O : Point := Unit.unit -- Center of the circle
def A : Point := Unit.unit
def B : Point := Unit.unit
def C : Point := Unit.unit
def D : Point := Unit.unit
def E : Point := Unit.unit

-- Define the radius of the circle
def radius : ℝ := 10

-- Define the necessary functions and properties
def isOnCircle (p : Point) : Prop := sorry
def isChord (p q : Point) : Prop := sorry
def isDiameter (p q : Point) : Prop := sorry
def intersectsAt (l1 l2 : Point × Point) (p : Point) : Prop := sorry
def distance (p q : Point) : ℝ := sorry
def angle (p q r : Point) : ℝ := sorry

-- State the theorem
theorem chord_intersection_sum_of_squares 
  (h1 : isOnCircle A ∧ isOnCircle B ∧ isOnCircle C ∧ isOnCircle D)
  (h2 : isDiameter A B)
  (h3 : isChord C D)
  (h4 : intersectsAt (A, B) (C, D) E)
  (h5 : distance B E = 6)
  (h6 : angle A E C = 60) :
  (distance C E)^2 + (distance D E)^2 = 300 := by sorry

end NUMINAMATH_CALUDE_chord_intersection_sum_of_squares_l2456_245632


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l2456_245636

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- The angles are supplementary (sum to 180°)
  a / b = 4 / 5 →  -- The ratio of the angles is 4:5
  b = 100 :=  -- The larger angle is 100°
by
  sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l2456_245636


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l2456_245699

theorem circle_area_from_circumference :
  ∀ (C : ℝ) (r : ℝ) (A : ℝ),
  C = 36 →
  C = 2 * π * r →
  A = π * r^2 →
  A = 324 / π := by
sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l2456_245699


namespace NUMINAMATH_CALUDE_minimum_square_formation_l2456_245634

theorem minimum_square_formation :
  ∃ (n : ℕ), 
    (∃ (k : ℕ), n = k^2) ∧ 
    (∃ (m : ℕ), 11*n + 1 = m^2) ∧
    (∀ (x : ℕ), x < n → ¬(∃ (j : ℕ), x = j^2) ∨ ¬(∃ (l : ℕ), 11*x + 1 = l^2)) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_minimum_square_formation_l2456_245634


namespace NUMINAMATH_CALUDE_largest_n_for_product_l2456_245601

/-- An arithmetic sequence with integer terms -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem largest_n_for_product (a b : ℕ → ℤ) :
  ArithmeticSequence a →
  ArithmeticSequence b →
  a 1 = 1 →
  b 1 = 1 →
  a 2 ≤ b 2 →
  (∃ n : ℕ, a n * b n = 1764) →
  (∀ m : ℕ, (∃ k : ℕ, a k * b k = 1764) → m ≤ 44) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_product_l2456_245601


namespace NUMINAMATH_CALUDE_f_properties_l2456_245681

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 
  Real.sqrt 3 * Real.sin (ω * x + φ) - Real.cos (ω * x + φ)

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def adjacentSymmetryDistance (f : ℝ → ℝ) (d : ℝ) : Prop :=
  ∀ x, f (x + d) = f x

theorem f_properties (ω φ : ℝ) 
  (h_φ : -π/2 < φ ∧ φ < 0) 
  (h_ω : ω > 0) 
  (h_even : isEven (f ω φ))
  (h_symmetry : adjacentSymmetryDistance (f ω φ) (π/2)) :
  f ω φ (π/24) = -(Real.sqrt 6 + Real.sqrt 2)/2 ∧
  ∃ g : ℝ → ℝ, g = fun x ↦ -2 * Real.cos (x/2 - π/3) := by
sorry

end NUMINAMATH_CALUDE_f_properties_l2456_245681


namespace NUMINAMATH_CALUDE_difference_p_q_l2456_245691

theorem difference_p_q (p q : ℚ) (hp : 3 / p = 8) (hq : 3 / q = 18) : p - q = 5 / 24 := by
  sorry

end NUMINAMATH_CALUDE_difference_p_q_l2456_245691


namespace NUMINAMATH_CALUDE_min_square_sum_on_line_l2456_245643

/-- The minimum value of x^2 + y^2 for points on the line x + y - 4 = 0 is 8 -/
theorem min_square_sum_on_line :
  ∃ (min : ℝ), min = 8 ∧
  ∀ (x y : ℝ), x + y - 4 = 0 →
  x^2 + y^2 ≥ min ∧
  ∃ (x₀ y₀ : ℝ), x₀ + y₀ - 4 = 0 ∧ x₀^2 + y₀^2 = min :=
by sorry

end NUMINAMATH_CALUDE_min_square_sum_on_line_l2456_245643


namespace NUMINAMATH_CALUDE_can_achieve_any_coloring_can_achieve_checkerboard_l2456_245649

/-- Represents a square on the chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the color of a square -/
inductive Color
  | White
  | Black

/-- Represents the state of the chessboard -/
def Board := Square → Color

/-- Represents a move that changes the color of squares in a row and column -/
structure Move where
  row : Fin 8
  col : Fin 8

/-- Applies a move to a board, changing colors in the specified row and column -/
def applyMove (b : Board) (m : Move) : Board :=
  fun s => if s.row = m.row || s.col = m.col then
             match b s with
             | Color.White => Color.Black
             | Color.Black => Color.White
           else b s

/-- The initial all-white board -/
def initialBoard : Board := fun _ => Color.White

/-- The standard checkerboard pattern -/
def checkerboardPattern : Board :=
  fun s => if (s.row.val + s.col.val) % 2 = 0 then Color.White else Color.Black

/-- Theorem stating that any desired board coloring can be achieved -/
theorem can_achieve_any_coloring :
  ∀ (targetBoard : Board), ∃ (moves : List Move),
    (moves.foldl applyMove initialBoard) = targetBoard :=
  sorry

/-- Corollary stating that the standard checkerboard pattern can be achieved -/
theorem can_achieve_checkerboard :
  ∃ (moves : List Move),
    (moves.foldl applyMove initialBoard) = checkerboardPattern :=
  sorry

end NUMINAMATH_CALUDE_can_achieve_any_coloring_can_achieve_checkerboard_l2456_245649


namespace NUMINAMATH_CALUDE_hyperbola_condition_l2456_245654

/-- Represents a curve defined by the equation x²/(4-t) + y²/(t-1) = 1 --/
def C (t : ℝ) := {(x, y) : ℝ × ℝ | x^2 / (4 - t) + y^2 / (t - 1) = 1}

/-- Defines when C is a hyperbola --/
def is_hyperbola (t : ℝ) : Prop := (4 - t) * (t - 1) < 0

/-- Theorem stating that C is a hyperbola iff t > 4 or t < 1 --/
theorem hyperbola_condition (t : ℝ) : 
  is_hyperbola t ↔ t > 4 ∨ t < 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l2456_245654


namespace NUMINAMATH_CALUDE_second_player_winning_strategy_l2456_245695

/-- Represents a node in the hexagonal grid --/
structure Node :=
  (x : ℤ)
  (y : ℤ)
  (z : ℤ)
  (sum_zero : x + y + z = 0)

/-- Represents the hexagonal grid game --/
structure HexagonGame :=
  (n : ℕ)
  (current_player : ℕ)
  (token : Node)
  (visited : Set Node)

/-- Defines a valid move in the game --/
def valid_move (game : HexagonGame) (new_pos : Node) : Prop :=
  (abs (new_pos.x - game.token.x) + abs (new_pos.y - game.token.y) + abs (new_pos.z - game.token.z) = 2) ∧
  (new_pos ∉ game.visited)

/-- Defines the winning condition for the second player --/
def second_player_wins (n : ℕ) : Prop :=
  ∀ (game : HexagonGame),
    game.n = n →
    (game.current_player = 1 → ∃ (move : Node), valid_move game move) →
    (game.current_player = 2 → ∀ (move : Node), valid_move game move → 
      ∃ (counter_move : Node), valid_move (HexagonGame.mk n 1 move (game.visited.insert game.token)) counter_move)

/-- The main theorem: The second player has a winning strategy for all n --/
theorem second_player_winning_strategy :
  ∀ n : ℕ, second_player_wins n :=
sorry

end NUMINAMATH_CALUDE_second_player_winning_strategy_l2456_245695


namespace NUMINAMATH_CALUDE_parallel_lines_count_l2456_245679

/-- Given two sets of intersecting parallel lines in a plane, where one set has 8 lines
    and the intersection forms 588 parallelograms, prove that the other set has 85 lines. -/
theorem parallel_lines_count (n : ℕ) 
  (h1 : n > 0)
  (h2 : (n - 1) * 7 = 588) : 
  n = 85 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_count_l2456_245679


namespace NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_l2456_245614

-- Define the types for lines and angles
variable (Line : Type) (Angle : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (corresponding_angles : Line → Line → Angle → Angle → Prop)
variable (equal_angles : Angle → Angle → Prop)

-- State the theorem
theorem parallel_lines_corresponding_angles 
  (l1 l2 : Line) (a1 a2 : Angle) : 
  (parallel l1 l2 → corresponding_angles l1 l2 a1 a2 → equal_angles a1 a2) ∧
  (corresponding_angles l1 l2 a1 a2 → equal_angles a1 a2 → parallel l1 l2) ∧
  (¬parallel l1 l2 → corresponding_angles l1 l2 a1 a2 → ¬equal_angles a1 a2) ∧
  (corresponding_angles l1 l2 a1 a2 → ¬equal_angles a1 a2 → ¬parallel l1 l2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_l2456_245614


namespace NUMINAMATH_CALUDE_peter_wants_17_dogs_l2456_245678

/-- The number of dogs Peter wants to have -/
def PetersDogs (samGS : ℕ) (samFB : ℕ) (peterGSFactor : ℕ) (peterFBFactor : ℕ) : ℕ :=
  peterGSFactor * samGS + peterFBFactor * samFB

/-- Theorem stating the number of dogs Peter wants to have -/
theorem peter_wants_17_dogs :
  PetersDogs 3 4 3 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_peter_wants_17_dogs_l2456_245678


namespace NUMINAMATH_CALUDE_optimal_price_achieves_target_profit_l2456_245641

/-- Represents the sales data and profit target for a fruit supermarket --/
structure FruitSales where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_reduction : ℝ
  sales_increase : ℝ
  target_profit : ℝ

/-- Calculates the optimal selling price for the fruit --/
def optimal_selling_price (data : FruitSales) : ℝ :=
  data.initial_price - (data.price_reduction * 3)

/-- Theorem stating that the optimal selling price achieves the target profit --/
theorem optimal_price_achieves_target_profit (data : FruitSales) 
  (h1 : data.cost_price = 22)
  (h2 : data.initial_price = 38)
  (h3 : data.initial_sales = 160)
  (h4 : data.price_reduction = 3)
  (h5 : data.sales_increase = 120)
  (h6 : data.target_profit = 3640) :
  let price := optimal_selling_price data
  let sales := data.initial_sales + data.sales_increase
  let profit_per_kg := price - data.cost_price
  profit_per_kg * sales = data.target_profit ∧ 
  price = 29 :=
by sorry

#eval optimal_selling_price { 
  cost_price := 22, 
  initial_price := 38, 
  initial_sales := 160, 
  price_reduction := 3, 
  sales_increase := 120, 
  target_profit := 3640 
}

end NUMINAMATH_CALUDE_optimal_price_achieves_target_profit_l2456_245641


namespace NUMINAMATH_CALUDE_f_decreasing_interval_f_max_value_l2456_245640

-- Define the function f(x) = x^3 - 3x^2
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Theorem for the decreasing interval
theorem f_decreasing_interval :
  ∀ x ∈ (Set.Ioo 0 2), ∀ y ∈ (Set.Ioo 0 2), x < y → f x > f y :=
sorry

-- Theorem for the maximum value on [-4, 3]
theorem f_max_value :
  ∀ x ∈ (Set.Icc (-4) 3), f x ≤ 0 ∧ ∃ y ∈ (Set.Icc (-4) 3), f y = 0 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_f_max_value_l2456_245640


namespace NUMINAMATH_CALUDE_equation_solution_l2456_245655

theorem equation_solution : 
  ∃! (x : ℝ), x > 0 ∧ (1/2) * (3*x^2 - 1) = (x^2 - 50*x - 10) * (x^2 + 25*x + 5) ∧ x = 25 + 2 * Real.sqrt 159 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2456_245655


namespace NUMINAMATH_CALUDE_press_conference_seating_l2456_245648

/-- Represents the number of ways to seat players from different teams -/
def seating_arrangements (cubs : Nat) (red_sox : Nat) : Nat :=
  2 * 2 * (Nat.factorial cubs) * (Nat.factorial red_sox)

/-- Theorem stating the number of seating arrangements for the given conditions -/
theorem press_conference_seating :
  seating_arrangements 4 3 = 576 :=
by sorry

end NUMINAMATH_CALUDE_press_conference_seating_l2456_245648


namespace NUMINAMATH_CALUDE_tuesday_sales_total_l2456_245672

/-- Represents the types of flowers sold in the shop -/
inductive FlowerType
  | Rose
  | Lilac
  | Gardenia
  | Tulip
  | Orchid

/-- Represents the sales data for a given day -/
structure SalesData where
  roses : ℕ
  lilacs : ℕ
  gardenias : ℕ
  tulips : ℕ
  orchids : ℕ

/-- Calculate the total number of flowers sold -/
def totalFlowers (sales : SalesData) : ℕ :=
  sales.roses + sales.lilacs + sales.gardenias + sales.tulips + sales.orchids

/-- Apply Tuesday sales factors to Monday's sales -/
def applyTuesdayFactors (monday : SalesData) : SalesData :=
  { roses := monday.roses - monday.roses * 4 / 100,
    lilacs := monday.lilacs + monday.lilacs * 5 / 100,
    gardenias := monday.gardenias,
    tulips := monday.tulips - monday.tulips * 7 / 100,
    orchids := monday.orchids }

/-- Theorem: Given the conditions, the total number of flowers sold on Tuesday is 214 -/
theorem tuesday_sales_total (monday : SalesData)
  (h1 : monday.lilacs = 15)
  (h2 : monday.roses = 3 * monday.lilacs)
  (h3 : monday.gardenias = monday.lilacs / 2)
  (h4 : monday.tulips = 2 * (monday.roses + monday.gardenias))
  (h5 : monday.orchids = (monday.roses + monday.gardenias + monday.tulips) / 3)
  : totalFlowers (applyTuesdayFactors monday) = 214 := by
  sorry


end NUMINAMATH_CALUDE_tuesday_sales_total_l2456_245672


namespace NUMINAMATH_CALUDE_max_d_value_l2456_245616

def a (n : ℕ) : ℕ := 150 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (k : ℕ), d k = 601 ∧ ∀ (n : ℕ), d n ≤ 601 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l2456_245616


namespace NUMINAMATH_CALUDE_train_length_l2456_245621

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 126 → time = 9 → speed * time * (1000 / 3600) = 315 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2456_245621


namespace NUMINAMATH_CALUDE_max_m_inequality_l2456_245673

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ m : ℝ, (3 / a + 1 / b ≥ m / (a + 3 * b)) → m ≤ 12) ∧
  ∃ m : ℝ, m = 12 ∧ (3 / a + 1 / b ≥ m / (a + 3 * b)) :=
by sorry

end NUMINAMATH_CALUDE_max_m_inequality_l2456_245673


namespace NUMINAMATH_CALUDE_incenter_inside_BOH_l2456_245600

/-- Triangle type with vertices A, B, C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Angle measure of a triangle -/
def angle (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

/-- Incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- Circumcenter of a triangle -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Check if a point is inside a triangle -/
def is_inside (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Main theorem: Incenter lies inside the triangle formed by circumcenter, vertex B, and orthocenter -/
theorem incenter_inside_BOH (t : Triangle) 
  (h1 : angle t t.C > angle t t.B)
  (h2 : angle t t.B > angle t t.A) : 
  is_inside (incenter t) (Triangle.mk (circumcenter t) t.B (orthocenter t)) := by
  sorry

end NUMINAMATH_CALUDE_incenter_inside_BOH_l2456_245600


namespace NUMINAMATH_CALUDE_grid_and_circles_area_sum_l2456_245618

/-- The side length of each small square in the grid -/
def smallSquareSide : ℝ := 3

/-- The number of rows in the grid -/
def gridRows : ℕ := 4

/-- The number of columns in the grid -/
def gridColumns : ℕ := 4

/-- The radius of the large circle -/
def largeCircleRadius : ℝ := 1.5 * smallSquareSide

/-- The radius of each small circle -/
def smallCircleRadius : ℝ := 0.5 * smallSquareSide

/-- The number of small circles -/
def numSmallCircles : ℕ := 3

/-- Theorem: The sum of the total grid area and the total area of the circles is 171 square cm -/
theorem grid_and_circles_area_sum : 
  (gridRows * gridColumns * smallSquareSide^2) + 
  (π * largeCircleRadius^2 + π * numSmallCircles * smallCircleRadius^2) = 171 := by
  sorry

end NUMINAMATH_CALUDE_grid_and_circles_area_sum_l2456_245618


namespace NUMINAMATH_CALUDE_trig_simplification_l2456_245623

theorem trig_simplification (α : ℝ) :
  Real.sin (α - 4 * Real.pi) * Real.sin (Real.pi - α) -
  2 * (Real.cos ((3 * Real.pi) / 2 + α))^2 -
  Real.sin (α + Real.pi) * Real.cos (Real.pi / 2 + α) =
  -2 * (Real.sin α)^2 := by sorry

end NUMINAMATH_CALUDE_trig_simplification_l2456_245623


namespace NUMINAMATH_CALUDE_john_sneezing_fit_duration_l2456_245612

/-- Calculates the duration of a sneezing fit given the time between sneezes and the number of sneezes. -/
def sneezingFitDuration (timeBetweenSneezes : ℕ) (numberOfSneezes : ℕ) : ℕ :=
  timeBetweenSneezes * numberOfSneezes

/-- Proves that a sneezing fit with 3 seconds between sneezes and 40 sneezes lasts 120 seconds. -/
theorem john_sneezing_fit_duration :
  sneezingFitDuration 3 40 = 120 := by
  sorry

#eval sneezingFitDuration 3 40

end NUMINAMATH_CALUDE_john_sneezing_fit_duration_l2456_245612


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2456_245686

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) + x * y = f x * f y) →
  (∀ x : ℝ, f x = 1 - x) ∨ (∀ x : ℝ, f x = x + 1) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2456_245686


namespace NUMINAMATH_CALUDE_center_is_five_l2456_245657

/-- Represents a 3x3 grid with numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if two positions in the grid are adjacent -/
def adjacent (p q : Fin 3 × Fin 3) : Prop :=
  (p.1 = q.1 ∧ p.2.succ = q.2) ∨
  (p.1 = q.1 ∧ p.2 = q.2.succ) ∨
  (p.1.succ = q.1 ∧ p.2 = q.2) ∨
  (p.1 = q.1.succ ∧ p.2 = q.2)

/-- Checks if two numbers are consecutive -/
def consecutive (m n : Fin 9) : Prop :=
  m.val + 1 = n.val ∨ n.val + 1 = m.val

/-- The main theorem -/
theorem center_is_five (g : Grid) :
  (∀ i j k l : Fin 3, i ≠ j → k ≠ l → g i k ≠ g j l) →  -- Each number is used once
  (∀ i j k l : Fin 3, consecutive (g i j) (g k l) → adjacent (i, j) (k, l)) →  -- Consecutive numbers are adjacent
  g 0 0 = 1 ∧ g 0 2 = 3 ∧ g 2 0 = 5 ∧ g 2 2 = 7 →  -- Corners are 2, 4, 6, 8
  g 1 1 = 4  -- Center is 5
  := by sorry

end NUMINAMATH_CALUDE_center_is_five_l2456_245657


namespace NUMINAMATH_CALUDE_max_ab_value_l2456_245661

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b + a + b = 1) :
  a * b ≤ 3 - 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_ab_value_l2456_245661


namespace NUMINAMATH_CALUDE_ax5_plus_by5_exists_l2456_245650

theorem ax5_plus_by5_exists (a b x y : ℝ) 
  (h1 : a*x + b*y = 4)
  (h2 : a*x^2 + b*y^2 = 10)
  (h3 : a*x^3 + b*y^3 = 28)
  (h4 : a*x^4 + b*y^4 = 82) :
  ∃ s5 : ℝ, a*x^5 + b*y^5 = s5 :=
by
  sorry

end NUMINAMATH_CALUDE_ax5_plus_by5_exists_l2456_245650


namespace NUMINAMATH_CALUDE_complex_real_condition_l2456_245626

theorem complex_real_condition (m : ℝ) : 
  let z : ℂ := (2 + m * Complex.I) / (1 + Complex.I)
  (z.im = 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l2456_245626


namespace NUMINAMATH_CALUDE_root_of_polynomial_l2456_245682

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 16*x^2 + 4

-- State the theorem
theorem root_of_polynomial :
  -- The polynomial is monic
  (∀ x, p x = x^4 - 16*x^2 + 4) ∧
  -- The polynomial has degree 4
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- √3 + √5 is a root of the polynomial
  p (Real.sqrt 3 + Real.sqrt 5) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_of_polynomial_l2456_245682


namespace NUMINAMATH_CALUDE_characterization_of_finite_sets_l2456_245662

def ClosedUnderAbsoluteSum (X : Set ℝ) : Prop :=
  ∀ x ∈ X, x + |x| ∈ X

theorem characterization_of_finite_sets (X : Set ℝ) 
  (h_nonempty : X.Nonempty) (h_finite : X.Finite) (h_closed : ClosedUnderAbsoluteSum X) :
  ∃ F : Set ℝ, F.Finite ∧ (∀ x ∈ F, x < 0) ∧ X = F ∪ {0} :=
sorry

end NUMINAMATH_CALUDE_characterization_of_finite_sets_l2456_245662


namespace NUMINAMATH_CALUDE_sum_mod_nine_l2456_245660

theorem sum_mod_nine : (3612 + 3613 + 3614 + 3615 + 3616) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l2456_245660


namespace NUMINAMATH_CALUDE_ages_ratio_three_to_one_l2456_245666

/-- Represents a person's age --/
structure Age where
  years : ℕ

/-- Represents the ages of Claire and Pete --/
structure AgesPair where
  claire : Age
  pete : Age

/-- The conditions of the problem --/
def problem_conditions (ages : AgesPair) : Prop :=
  (ages.claire.years - 3 = 2 * (ages.pete.years - 3)) ∧
  (ages.pete.years - 7 = (ages.claire.years - 7) / 4)

/-- The theorem to prove --/
theorem ages_ratio_three_to_one (ages : AgesPair) :
  problem_conditions ages →
  ∃ (claire_age pete_age : ℕ),
    claire_age = ages.claire.years - 6 ∧
    pete_age = ages.pete.years - 6 ∧
    3 * pete_age = claire_age :=
by
  sorry


end NUMINAMATH_CALUDE_ages_ratio_three_to_one_l2456_245666


namespace NUMINAMATH_CALUDE_helen_gas_consumption_l2456_245602

/-- Represents the gas consumption for Helen's lawn maintenance --/
def lawn_maintenance_gas_consumption : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ := sorry

/-- The number of times the large lawn is cut --/
def large_lawn_cuts : ℕ := 18

/-- The number of times the small lawn is cut --/
def small_lawn_cuts : ℕ := 14

/-- The number of times the suburban lawn is trimmed --/
def suburban_trims : ℕ := 6

/-- The number of times the leaf blower is used --/
def leaf_blower_uses : ℕ := 2

theorem helen_gas_consumption :
  lawn_maintenance_gas_consumption large_lawn_cuts small_lawn_cuts suburban_trims leaf_blower_uses 3 2 = 22 := by sorry

end NUMINAMATH_CALUDE_helen_gas_consumption_l2456_245602


namespace NUMINAMATH_CALUDE_parabola_line_intersection_sum_l2456_245674

/-- Parabola P with equation y = x^2 -/
def P : ℝ → ℝ := fun x ↦ x^2

/-- Point Q -/
def Q : ℝ × ℝ := (10, -6)

/-- Line through Q with slope m -/
def line_through_Q (m : ℝ) : ℝ → ℝ := fun x ↦ m * (x - Q.1) + Q.2

/-- The line does not intersect the parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line_through_Q m x

/-- Theorem stating that r + s = 40 -/
theorem parabola_line_intersection_sum :
  ∃ r s : ℝ, (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) → r + s = 40 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_sum_l2456_245674


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_31_l2456_245631

theorem smallest_n_divisible_by_31 :
  ∃ (n : ℕ), n > 0 ∧ (31 ∣ (5^n + n)) ∧ ∀ (m : ℕ), m > 0 ∧ (31 ∣ (5^m + m)) → n ≤ m :=
by
  use 30
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_31_l2456_245631


namespace NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_l2456_245603

def repeating_decimal_to_fraction (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 990 + a / 10

theorem repeating_decimal_equiv_fraction :
  repeating_decimal_to_fraction 2 1 3 = 523 / 2475 ∧
  (∀ m n : ℕ, m ≠ 0 → n ≠ 0 → m / n = 523 / 2475 → m ≥ 523 ∧ n ≥ 2475) :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_l2456_245603


namespace NUMINAMATH_CALUDE_three_numbers_sum_square_counterexample_l2456_245675

theorem three_numbers_sum_square_counterexample :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a + b^2 + c^2 = b + a^2 + c^2) ∧
    (b + a^2 + c^2 = c + a^2 + b^2) ∧
    (a ≠ b ∨ b ≠ c ∨ a ≠ c) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_sum_square_counterexample_l2456_245675


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l2456_245610

/-- Given a triangle with sides in ratio 1/2 : 1/3 : 1/4 and perimeter 104 cm, 
    the longest side is 48 cm. -/
theorem longest_side_of_triangle (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → -- sides are positive
  a / b = 3 / 2 ∧ b / c = 4 / 3 → -- ratio condition
  a + b + c = 104 → -- perimeter condition
  a = 48 := by sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l2456_245610


namespace NUMINAMATH_CALUDE_simplify_sqrt_product_simplify_log_product_l2456_245669

-- Part I
theorem simplify_sqrt_product (a : ℝ) (ha : 0 < a) :
  Real.sqrt (a^(1/4)) * Real.sqrt (a * Real.sqrt a) = Real.sqrt a := by sorry

-- Part II
theorem simplify_log_product :
  Real.log 3 / Real.log 2 * Real.log 5 / Real.log 3 * Real.log 4 / Real.log 5 = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_simplify_log_product_l2456_245669


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l2456_245606

theorem geometric_progression_first_term (S a r : ℝ) : 
  S = 10 → 
  a + a * r = 6 → 
  a = 2 * r → 
  (a = -1 + Real.sqrt 13 ∨ a = -1 - Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l2456_245606


namespace NUMINAMATH_CALUDE_hyperbola_parabola_focus_l2456_245637

/-- The value of 'a' for a hyperbola with equation x^2 - y^2 = a^2 (a > 0) 
    whose right focus coincides with the focus of the parabola y^2 = 4x -/
theorem hyperbola_parabola_focus (a : ℝ) : a > 0 → 
  (∃ (x y : ℝ), x^2 - y^2 = a^2 ∧ y^2 = 4*x ∧ (x, y) = (1, 0)) → 
  a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_focus_l2456_245637


namespace NUMINAMATH_CALUDE_complex_sum_equals_negative_two_l2456_245667

theorem complex_sum_equals_negative_two (w : ℂ) : 
  w = Complex.cos (3 * Real.pi / 8) + Complex.I * Complex.sin (3 * Real.pi / 8) →
  2 * (w / (1 + w^3) + w^2 / (1 + w^6) + w^3 / (1 + w^9)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_negative_two_l2456_245667


namespace NUMINAMATH_CALUDE_unique_perfect_square_in_range_l2456_245628

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y * y

theorem unique_perfect_square_in_range :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 14 →
    (is_perfect_square (n.factorial * (n + 1).factorial / 3) ↔ n = 11) :=
by sorry

end NUMINAMATH_CALUDE_unique_perfect_square_in_range_l2456_245628


namespace NUMINAMATH_CALUDE_megan_folders_l2456_245664

/-- The number of folders Megan ended up with -/
def num_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : ℕ :=
  (initial_files - deleted_files) / files_per_folder

/-- Proof that Megan ended up with 9 folders -/
theorem megan_folders : num_folders 93 21 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_megan_folders_l2456_245664


namespace NUMINAMATH_CALUDE_triangle_side_length_l2456_245611

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h_area : (1/2) * a * c * (Real.sqrt 3 / 2) = Real.sqrt 3)
  (h_angle : B = 60 * π / 180)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2456_245611


namespace NUMINAMATH_CALUDE_player_in_first_and_last_game_l2456_245619

/-- Represents a chess tournament. -/
structure ChessTournament (n : ℕ) where
  /-- The number of players in the tournament. -/
  num_players : ℕ
  /-- The total number of games played in the tournament. -/
  num_games : ℕ
  /-- Condition that the number of players is 2n+3. -/
  player_count : num_players = 2*n + 3
  /-- Condition that the number of games is (2n+3)*(2n+2)/2. -/
  game_count : num_games = (num_players * (num_players - 1)) / 2
  /-- Function that returns true if a player played in a specific game. -/
  played_in_game : ℕ → ℕ → Prop
  /-- Condition that each player rests for at least n games after each match. -/
  rest_condition : ∀ p g₁ g₂, played_in_game p g₁ → played_in_game p g₂ → g₁ < g₂ → g₂ - g₁ > n

/-- Theorem stating that a player who played in the first game also played in the last game. -/
theorem player_in_first_and_last_game (n : ℕ) (tournament : ChessTournament n) :
  ∃ p, tournament.played_in_game p 1 ∧ tournament.played_in_game p tournament.num_games :=
sorry

end NUMINAMATH_CALUDE_player_in_first_and_last_game_l2456_245619


namespace NUMINAMATH_CALUDE_travel_distances_l2456_245646

-- Define the given constants
def train_speed : ℚ := 100
def car_speed_ratio : ℚ := 2/3
def bicycle_speed_ratio : ℚ := 1/5
def travel_time : ℚ := 1/2  -- 30 minutes in hours

-- Define the theorem
theorem travel_distances :
  let car_distance := train_speed * car_speed_ratio * travel_time
  let bicycle_distance := train_speed * bicycle_speed_ratio * travel_time
  car_distance = 100/3 ∧ bicycle_distance = 10 := by sorry

end NUMINAMATH_CALUDE_travel_distances_l2456_245646


namespace NUMINAMATH_CALUDE_oak_grove_library_books_l2456_245608

/-- The number of books in Oak Grove's public library -/
def public_library_books : ℕ := 1986

/-- The number of books in Oak Grove's school libraries -/
def school_library_books : ℕ := 5106

/-- The total number of books in Oak Grove libraries -/
def total_books : ℕ := public_library_books + school_library_books

theorem oak_grove_library_books : total_books = 7092 := by
  sorry

end NUMINAMATH_CALUDE_oak_grove_library_books_l2456_245608


namespace NUMINAMATH_CALUDE_max_value_expression_l2456_245604

open Real

theorem max_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (⨆ x : ℝ, 2 * (a - x) * (x - Real.sqrt (x^2 + b^2))) = b^2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2456_245604


namespace NUMINAMATH_CALUDE_absolute_value_of_expression_l2456_245613

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem absolute_value_of_expression : 
  Complex.abs (2 + i^2 + 2*i^3) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_expression_l2456_245613


namespace NUMINAMATH_CALUDE_chess_tournament_wins_l2456_245607

theorem chess_tournament_wins (susan_wins susan_losses mike_wins mike_losses lana_losses : ℕ) 
  (h1 : susan_wins = 5)
  (h2 : susan_losses = 1)
  (h3 : mike_wins = 2)
  (h4 : mike_losses = 4)
  (h5 : lana_losses = 5)
  (h6 : susan_wins + mike_wins + lana_losses = susan_losses + mike_losses + lana_wins)
  : lana_wins = 3 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_wins_l2456_245607


namespace NUMINAMATH_CALUDE_sum_of_digits_square_l2456_245696

/-- Sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ :=
  sorry

/-- Theorem: A positive integer equals the square of the sum of its digits if and only if it's 1 or 81 -/
theorem sum_of_digits_square (n : ℕ+) : n = (sum_of_digits n)^2 ↔ n = 1 ∨ n = 81 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_square_l2456_245696


namespace NUMINAMATH_CALUDE_f_matches_table_l2456_245665

/-- The function that generates the output values -/
def f (n : ℕ) : ℕ := 2 * n - 1

/-- The proposition that the function f matches the given table for n from 1 to 5 -/
theorem f_matches_table : 
  f 1 = 1 ∧ f 2 = 3 ∧ f 3 = 5 ∧ f 4 = 7 ∧ f 5 = 9 := by
  sorry

#check f_matches_table

end NUMINAMATH_CALUDE_f_matches_table_l2456_245665


namespace NUMINAMATH_CALUDE_smallest_n_containing_all_binary_l2456_245633

/-- Given a natural number n, returns true if the binary representation of 1/n
    contains the binary representations of all numbers from 1 to 1990 as
    contiguous substrings after the decimal point. -/
def containsAllBinaryRepresentations (n : ℕ) : Prop := sorry

/-- Theorem stating that 2053 is the smallest natural number satisfying
    the condition of containing all binary representations from 1 to 1990. -/
theorem smallest_n_containing_all_binary : ∀ n : ℕ,
  n < 2053 → ¬(containsAllBinaryRepresentations n) ∧ containsAllBinaryRepresentations 2053 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_containing_all_binary_l2456_245633


namespace NUMINAMATH_CALUDE_pages_to_read_thursday_l2456_245688

def book_pages : ℕ := 158
def monday_pages : ℕ := 23
def tuesday_pages : ℕ := 38
def wednesday_pages : ℕ := 61

theorem pages_to_read_thursday (thursday_pages : ℕ) : 
  thursday_pages = 12 ↔ 
  ∃ (friday_pages : ℕ),
    friday_pages = 2 * thursday_pages ∧
    monday_pages + tuesday_pages + wednesday_pages + thursday_pages + friday_pages = book_pages :=
by sorry

end NUMINAMATH_CALUDE_pages_to_read_thursday_l2456_245688


namespace NUMINAMATH_CALUDE_intersection_M_N_l2456_245645

-- Define the sets M and N
def M : Set ℝ := {x | x ≤ 4}
def N : Set ℝ := {x | x > 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x ≤ 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2456_245645


namespace NUMINAMATH_CALUDE_mary_max_earnings_l2456_245625

/-- Calculates the maximum weekly earnings for a worker with the given conditions --/
def maxWeeklyEarnings (maxHours : ℕ) (regularHours : ℕ) (regularRate : ℚ) (overtimeRateIncrease : ℚ) : ℚ :=
  let overtimeHours := maxHours - regularHours
  let overtimeRate := regularRate * (1 + overtimeRateIncrease)
  regularHours * regularRate + overtimeHours * overtimeRate

/-- Theorem stating that Mary's maximum weekly earnings are $410 --/
theorem mary_max_earnings :
  maxWeeklyEarnings 45 20 8 (1/4) = 410 := by
  sorry

#eval maxWeeklyEarnings 45 20 8 (1/4)

end NUMINAMATH_CALUDE_mary_max_earnings_l2456_245625


namespace NUMINAMATH_CALUDE_unique_solution_l2456_245689

theorem unique_solution (a b c : ℝ) 
  (ha : a > 4) (hb : b > 4) (hc : c > 4)
  (heq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  a = 12 ∧ b = 10 ∧ c = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2456_245689


namespace NUMINAMATH_CALUDE_minimum_parts_to_exceed_plan_l2456_245690

def plan : ℕ := 40
def excess_percentage : ℚ := 47/100

theorem minimum_parts_to_exceed_plan : 
  ∀ n : ℕ, (n : ℚ) ≥ plan * (1 + excess_percentage) → n ≥ 59 :=
sorry

end NUMINAMATH_CALUDE_minimum_parts_to_exceed_plan_l2456_245690


namespace NUMINAMATH_CALUDE_vector_magnitude_l2456_245627

theorem vector_magnitude (a b : ℝ × ℝ) 
  (h1 : ‖b‖ = 5)
  (h2 : ‖2 • a + b‖ = 5 * Real.sqrt 3)
  (h3 : ‖a - b‖ = 5 * Real.sqrt 2) :
  ‖a‖ = 5 * Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l2456_245627


namespace NUMINAMATH_CALUDE_congruence_mod_nine_l2456_245635

theorem congruence_mod_nine : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_congruence_mod_nine_l2456_245635


namespace NUMINAMATH_CALUDE_percent_decrease_proof_l2456_245698

theorem percent_decrease_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 30) : 
  (original_price - sale_price) / original_price * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_proof_l2456_245698


namespace NUMINAMATH_CALUDE_min_sum_squares_l2456_245687

def S : Finset Int := {-6, -4, -3, -1, 1, 3, 5, 8}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (∀ a b c d e f g h : Int, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h →
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S →
    (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 5) ∧
  (p + q + r + s)^2 + (t + u + v + w)^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2456_245687


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l2456_245663

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem first_term_of_arithmetic_sequence :
  ∃ a₁ : ℤ, arithmetic_sequence a₁ 2 15 = -10 ∧ a₁ = -38 := by sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l2456_245663


namespace NUMINAMATH_CALUDE_student_age_ratio_l2456_245651

/-- Represents the number of students in different age groups -/
structure SchoolPopulation where
  total : ℕ
  below_eight : ℕ
  eight_years : ℕ
  above_eight : ℕ

/-- Theorem stating the ratio of students above 8 years to 8 years old -/
theorem student_age_ratio (school : SchoolPopulation) 
  (h1 : school.total = 80)
  (h2 : school.below_eight = school.total / 4)
  (h3 : school.eight_years = 36)
  (h4 : school.above_eight = school.total - school.below_eight - school.eight_years) :
  (school.above_eight : ℚ) / school.eight_years = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_student_age_ratio_l2456_245651


namespace NUMINAMATH_CALUDE_toothpicks_10th_stage_l2456_245620

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 5
  else toothpicks (n - 1) + 3 * n

/-- Theorem: The number of toothpicks in the 10th stage is 167 -/
theorem toothpicks_10th_stage :
  toothpicks 10 = 167 := by
  sorry

#eval toothpicks 10  -- For verification

end NUMINAMATH_CALUDE_toothpicks_10th_stage_l2456_245620


namespace NUMINAMATH_CALUDE_three_digit_divisibility_l2456_245692

theorem three_digit_divisibility (a b c : ℕ) (p : ℕ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0)
  (h_p : Nat.Prime p) (h_abc : p ∣ (100 * a + 10 * b + c)) (h_cba : p ∣ (100 * c + 10 * b + a)) :
  p ∣ (a + b + c) ∨ p ∣ (a - b + c) ∨ p ∣ (a - c) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisibility_l2456_245692


namespace NUMINAMATH_CALUDE_specific_field_planted_fraction_l2456_245629

/-- Represents a right-angled triangular field with an unplanted square at the right angle. -/
structure TriangularField where
  leg1 : ℝ
  leg2 : ℝ
  square_distance : ℝ

/-- Calculates the fraction of the field that is planted. -/
def planted_fraction (field : TriangularField) : ℚ :=
  sorry

/-- Theorem stating that for a specific field configuration, the planted fraction is 7/10. -/
theorem specific_field_planted_fraction :
  let field : TriangularField := { leg1 := 5, leg2 := 12, square_distance := 3 }
  planted_fraction field = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_specific_field_planted_fraction_l2456_245629


namespace NUMINAMATH_CALUDE_expand_equals_difference_of_squares_l2456_245615

theorem expand_equals_difference_of_squares (x y : ℝ) :
  (-x + 2*y) * (-x - 2*y) = x^2 - 4*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_equals_difference_of_squares_l2456_245615


namespace NUMINAMATH_CALUDE_unused_types_count_l2456_245656

/-- The number of natural resources -/
def num_resources : ℕ := 6

/-- The number of types of nature use already developed -/
def developed_types : ℕ := 23

/-- The total number of possible combinations of resource usage -/
def total_combinations : ℕ := 2^num_resources

/-- The number of valid combinations (excluding the all-zero combination) -/
def valid_combinations : ℕ := total_combinations - 1

theorem unused_types_count : valid_combinations - developed_types = 40 := by
  sorry

end NUMINAMATH_CALUDE_unused_types_count_l2456_245656


namespace NUMINAMATH_CALUDE_area_of_APEG_l2456_245609

/-- Two squares with side lengths 8 and 6 placed side by side -/
structure TwoSquares where
  squareABCD : Set (ℝ × ℝ)
  squareBEFG : Set (ℝ × ℝ)
  sideAB : ℝ
  sideBE : ℝ
  B : ℝ × ℝ
  common_point : B ∈ squareABCD ∩ squareBEFG
  sideAB_length : sideAB = 8
  sideBE_length : sideBE = 6

/-- The quadrilateral APEG formed by the intersection of DE and BG -/
def quadrilateralAPEG (ts : TwoSquares) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The main theorem: The area of quadrilateral APEG is 18 -/
theorem area_of_APEG (ts : TwoSquares) : area (quadrilateralAPEG ts) = 18 :=
  sorry

end NUMINAMATH_CALUDE_area_of_APEG_l2456_245609


namespace NUMINAMATH_CALUDE_min_eating_time_is_23_5_l2456_245605

/-- Represents the eating rates and constraints for Amy and Ben -/
structure EatingProblem where
  total_carrots : ℕ
  total_muffins : ℕ
  wait_time : ℕ
  amy_carrot_rate : ℕ
  amy_muffin_rate : ℕ
  ben_carrot_rate : ℕ
  ben_muffin_rate : ℕ

/-- Calculates the minimum time to eat all food given the problem constraints -/
def min_eating_time (problem : EatingProblem) : ℚ :=
  sorry

/-- Theorem stating that the minimum eating time for the given problem is 23.5 minutes -/
theorem min_eating_time_is_23_5 : 
  let problem : EatingProblem := {
    total_carrots := 1000
    total_muffins := 1000
    wait_time := 5
    amy_carrot_rate := 40
    amy_muffin_rate := 70
    ben_carrot_rate := 60
    ben_muffin_rate := 30
  }
  min_eating_time problem = 47/2 := by
  sorry

end NUMINAMATH_CALUDE_min_eating_time_is_23_5_l2456_245605


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l2456_245693

theorem complex_magnitude_example : Complex.abs (-5 + (8/3)*Complex.I) = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l2456_245693


namespace NUMINAMATH_CALUDE_square_root_equation_l2456_245697

theorem square_root_equation (y : ℝ) : 
  Real.sqrt (9 + Real.sqrt (4 * y - 5)) = Real.sqrt 10 → y = (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l2456_245697


namespace NUMINAMATH_CALUDE_equal_roots_condition_l2456_245680

theorem equal_roots_condition (m : ℝ) : 
  (∃ (x : ℝ), (x * (x - 1) - (m + 3)) / ((x - 1) * (m - 1)) = x / m ∧ 
   (∀ (y : ℝ), (y * (y - 1) - (m + 3)) / ((y - 1) * (m - 1)) = y / m → y = x)) ↔ 
  (m = -1.5 + Real.sqrt 2 ∨ m = -1.5 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l2456_245680


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2456_245638

theorem problem_1 : (1) - 1/2 / 3 * (3 - (-3)^2) = 1 := by sorry

theorem problem_2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) : 
  2*x / (x^2 - 4) - 1 / (x - 2) = 1 / (x + 2) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2456_245638


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2456_245685

theorem trigonometric_equation_solution (k : ℤ) :
  (∃ x : ℝ, 2 + Real.cos x ^ 2 + Real.cos (4 * x) + Real.cos (2 * x) + 
   2 * Real.sin (3 * x) * Real.sin (7 * x) + Real.sin (7 * x) ^ 2 = 
   Real.cos (π * k / 2022) ^ 2) ↔ 
  (∃ m : ℤ, k = 2022 * m) ∧
  (∀ x : ℝ, 2 + Real.cos x ^ 2 + Real.cos (4 * x) + Real.cos (2 * x) + 
   2 * Real.sin (3 * x) * Real.sin (7 * x) + Real.sin (7 * x) ^ 2 = 
   Real.cos (π * k / 2022) ^ 2 →
   ∃ n : ℤ, x = π / 4 + π * n / 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2456_245685


namespace NUMINAMATH_CALUDE_triangle_constructible_l2456_245644

/-- Given a side length, angle bisector length, and altitude length of a triangle,
    prove that the triangle can be constructed uniquely if and only if
    the angle bisector length is greater than the altitude length. -/
theorem triangle_constructible (a f_a m_a : ℝ) (h_pos : a > 0 ∧ f_a > 0 ∧ m_a > 0) :
  ∃! (b c : ℝ), (b > 0 ∧ c > 0) ∧
    (∃ (α β γ : ℝ), 
      α > 0 ∧ β > 0 ∧ γ > 0 ∧
      α + β + γ = π ∧
      a^2 = b^2 + c^2 - 2*b*c*Real.cos α ∧
      f_a^2 = (b*c / (b + c))^2 + (a/2)^2 ∧
      m_a = a * Real.sin β / 2) ↔
  f_a > m_a :=
sorry

end NUMINAMATH_CALUDE_triangle_constructible_l2456_245644


namespace NUMINAMATH_CALUDE_fraction_integer_pairs_l2456_245653

theorem fraction_integer_pairs (m n : ℕ+) :
  (∃ h : ℕ+, (m.val^2 : ℚ) / (2 * m.val * n.val^2 - n.val^3 + 1) = h.val) ↔
  (∃ k : ℕ+, (m = 2 * k ∧ n = 1) ∨
             (m = k ∧ n = 2 * k) ∨
             (m = 8 * k.val^4 - k.val ∧ n = 2 * k)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_integer_pairs_l2456_245653


namespace NUMINAMATH_CALUDE_person_B_age_l2456_245668

theorem person_B_age 
  (avg_ABC : (age_A + age_B + age_C) / 3 = 22)
  (avg_AB : (age_A + age_B) / 2 = 18)
  (avg_BC : (age_B + age_C) / 2 = 25)
  : age_B = 20 := by
  sorry

end NUMINAMATH_CALUDE_person_B_age_l2456_245668


namespace NUMINAMATH_CALUDE_smoking_health_negative_correlation_l2456_245652

-- Define the type for relationships
inductive Relationship
| ParentChildHeight
| SmokingHealth
| CropYieldFertilization
| MathPhysicsGrades

-- Define a function to determine if a relationship is negatively correlated
def is_negatively_correlated (r : Relationship) : Prop :=
  match r with
  | Relationship.SmokingHealth => True
  | _ => False

-- Theorem statement
theorem smoking_health_negative_correlation :
  ∀ r : Relationship, is_negatively_correlated r ↔ r = Relationship.SmokingHealth :=
by sorry

end NUMINAMATH_CALUDE_smoking_health_negative_correlation_l2456_245652


namespace NUMINAMATH_CALUDE_colonization_combinations_count_l2456_245671

/-- Represents the number of Earth-like planets -/
def earth_like_planets : ℕ := 6

/-- Represents the number of Mars-like planets -/
def mars_like_planets : ℕ := 7

/-- Represents the resource cost to colonize an Earth-like planet -/
def earth_like_cost : ℕ := 2

/-- Represents the resource cost to colonize a Mars-like planet -/
def mars_like_cost : ℕ := 1

/-- Represents the total available resources -/
def total_resources : ℕ := 14

/-- Calculates the number of ways to select planets for colonization -/
def colonization_combinations : ℕ := sorry

/-- Theorem stating that the number of colonization combinations is 336 -/
theorem colonization_combinations_count :
  colonization_combinations = 336 := by sorry

end NUMINAMATH_CALUDE_colonization_combinations_count_l2456_245671


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2456_245639

def complex_number : ℂ := Complex.I * (1 - Complex.I)

theorem complex_number_in_first_quadrant : 
  complex_number.re > 0 ∧ complex_number.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2456_245639


namespace NUMINAMATH_CALUDE_probability_of_red_and_flag_in_three_draws_l2456_245659

/-- Represents a single draw from the bag -/
inductive Ball : Type
| wind : Ball
| exhibition : Ball
| red : Ball
| flag : Ball

/-- Represents a set of three draws -/
def DrawSet := (Ball × Ball × Ball)

/-- The sample data of 20 draw sets -/
def sampleData : List DrawSet := [
  (Ball.wind, Ball.red, Ball.red),
  (Ball.exhibition, Ball.flag, Ball.red),
  (Ball.flag, Ball.exhibition, Ball.wind),
  (Ball.wind, Ball.red, Ball.exhibition),
  (Ball.red, Ball.red, Ball.exhibition),
  (Ball.wind, Ball.wind, Ball.flag),
  (Ball.exhibition, Ball.red, Ball.flag),
  (Ball.red, Ball.wind, Ball.wind),
  (Ball.flag, Ball.flag, Ball.red),
  (Ball.red, Ball.exhibition, Ball.flag),
  (Ball.red, Ball.red, Ball.wind),
  (Ball.red, Ball.wind, Ball.exhibition),
  (Ball.red, Ball.red, Ball.red),
  (Ball.flag, Ball.wind, Ball.wind),
  (Ball.flag, Ball.red, Ball.exhibition),
  (Ball.flag, Ball.flag, Ball.wind),
  (Ball.exhibition, Ball.exhibition, Ball.flag),
  (Ball.red, Ball.exhibition, Ball.exhibition),
  (Ball.red, Ball.red, Ball.flag),
  (Ball.red, Ball.flag, Ball.flag)
]

/-- Checks if a draw set contains both red and flag balls -/
def containsRedAndFlag (s : DrawSet) : Bool :=
  match s with
  | (Ball.red, Ball.flag, _) | (Ball.red, _, Ball.flag) | (Ball.flag, Ball.red, _) 
  | (Ball.flag, _, Ball.red) | (_, Ball.red, Ball.flag) | (_, Ball.flag, Ball.red) => true
  | _ => false

/-- Counts the number of draw sets containing both red and flag balls -/
def countRedAndFlag (data : List DrawSet) : Nat :=
  data.filter containsRedAndFlag |>.length

/-- The theorem to be proved -/
theorem probability_of_red_and_flag_in_three_draws : 
  (countRedAndFlag sampleData : ℚ) / sampleData.length = 3 / 20 := by
  sorry


end NUMINAMATH_CALUDE_probability_of_red_and_flag_in_three_draws_l2456_245659


namespace NUMINAMATH_CALUDE_four_card_selection_ways_l2456_245630

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def num_suits : ℕ := 4

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := deck_size / num_suits

/-- The number of cards to choose -/
def cards_to_choose : ℕ := 4

/-- Theorem stating the number of ways to choose 4 cards from a standard deck
    with exactly two of the same suit and the other two of different suits -/
theorem four_card_selection_ways :
  (num_suits.choose 1) *
  ((num_suits - 1).choose 2) *
  (cards_per_suit.choose 2) *
  (cards_per_suit ^ 2) = 158004 := by
  sorry

end NUMINAMATH_CALUDE_four_card_selection_ways_l2456_245630


namespace NUMINAMATH_CALUDE_grocery_cost_l2456_245670

/-- The cost of groceries problem -/
theorem grocery_cost (mango_cost rice_cost flour_cost : ℝ)
  (h1 : 10 * mango_cost = 24 * rice_cost)
  (h2 : flour_cost = 2 * rice_cost)
  (h3 : flour_cost = 23) :
  4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 260.90 := by
  sorry

end NUMINAMATH_CALUDE_grocery_cost_l2456_245670


namespace NUMINAMATH_CALUDE_range_of_m_l2456_245647

/-- Given conditions p and q, prove that m ∈ [4, +∞) -/
theorem range_of_m (x m : ℝ) : 
  (∀ x, x^2 - 3*x - 4 ≤ 0 → |x - 3| ≤ m) ∧ 
  (∃ x, |x - 3| ≤ m ∧ x^2 - 3*x - 4 > 0) →
  m ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l2456_245647


namespace NUMINAMATH_CALUDE_pink_crayons_count_l2456_245676

def total_crayons : ℕ := 24
def red_crayons : ℕ := 8
def blue_crayons : ℕ := 6
def green_crayons : ℕ := (2 * blue_crayons) / 3

theorem pink_crayons_count :
  total_crayons - red_crayons - blue_crayons - green_crayons = 6 := by
  sorry

end NUMINAMATH_CALUDE_pink_crayons_count_l2456_245676


namespace NUMINAMATH_CALUDE_chess_matches_l2456_245617

theorem chess_matches (n : ℕ) (m : ℕ) (h1 : n = 5) (h2 : m = 3) :
  (n * (n - 1) * m) / 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_chess_matches_l2456_245617


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2456_245642

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 4 / b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 1 / a₀ + 4 / b₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2456_245642


namespace NUMINAMATH_CALUDE_total_dresses_l2456_245684

theorem total_dresses (ana_dresses : ℕ) (lisa_more_dresses : ℕ) : 
  ana_dresses = 15 → lisa_more_dresses = 18 → 
  ana_dresses + (ana_dresses + lisa_more_dresses) = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_dresses_l2456_245684


namespace NUMINAMATH_CALUDE_alchemerion_age_proof_l2456_245622

/-- Alchemerion's age in years -/
def alchemerion_age : ℕ := 360

/-- Alchemerion's son's age in years -/
def son_age : ℕ := alchemerion_age / 3

/-- Alchemerion's father's age in years -/
def father_age : ℕ := 2 * alchemerion_age + 40

theorem alchemerion_age_proof :
  (alchemerion_age = 3 * son_age) ∧
  (father_age = 2 * alchemerion_age + 40) ∧
  (alchemerion_age + son_age + father_age = 1240) :=
by sorry

end NUMINAMATH_CALUDE_alchemerion_age_proof_l2456_245622


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2456_245677

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_roots : a 3 * a 7 = 4 ∧ a 3 + a 7 = 5) :
  a 5 = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2456_245677


namespace NUMINAMATH_CALUDE_exists_permutation_9_not_exists_permutation_11_exists_permutation_1996_l2456_245694

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- Theorem for n = 9 -/
theorem exists_permutation_9 :
  ∃ f : Fin 9 → Fin 9, Function.Bijective f ∧
    ∀ k : Fin 9, isPerfectSquare ((k : ℕ) + 1 + (f k : ℕ)) :=
sorry

/-- Theorem for n = 11 -/
theorem not_exists_permutation_11 :
  ¬ ∃ f : Fin 11 → Fin 11, Function.Bijective f ∧
    ∀ k : Fin 11, isPerfectSquare ((k : ℕ) + 1 + (f k : ℕ)) :=
sorry

/-- Theorem for n = 1996 -/
theorem exists_permutation_1996 :
  ∃ f : Fin 1996 → Fin 1996, Function.Bijective f ∧
    ∀ k : Fin 1996, isPerfectSquare ((k : ℕ) + 1 + (f k : ℕ)) :=
sorry

end NUMINAMATH_CALUDE_exists_permutation_9_not_exists_permutation_11_exists_permutation_1996_l2456_245694


namespace NUMINAMATH_CALUDE_tank_length_calculation_l2456_245658

/-- Given a rectangular field and a tank dug within it, this theorem proves
    the length of the tank when the excavated earth raises the field level. -/
theorem tank_length_calculation (field_length field_width tank_width tank_depth level_rise : ℝ)
  (h1 : field_length = 90)
  (h2 : field_width = 50)
  (h3 : tank_width = 20)
  (h4 : tank_depth = 4)
  (h5 : level_rise = 0.5)
  (h6 : tank_width < field_width)
  (h7 : ∀ tank_length, tank_length > 0 → tank_length < field_length) :
  ∃ tank_length : ℝ,
    tank_length > 0 ∧
    tank_length < field_length ∧
    tank_length * tank_width * tank_depth =
      (field_length * field_width - tank_length * tank_width) * level_rise ∧
    tank_length = 25 := by
  sorry


end NUMINAMATH_CALUDE_tank_length_calculation_l2456_245658


namespace NUMINAMATH_CALUDE_sets_theorem_l2456_245683

-- Define the sets A, B, and C
def A : Set ℝ := {x | -3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem sets_theorem (a : ℝ) :
  (A ∪ B = {x : ℝ | -3 ≤ x ∧ x < 10}) ∧
  ((Set.compl A) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10}) ∧
  ((A ∩ C a).Nonempty ↔ a > -3) :=
by sorry

end NUMINAMATH_CALUDE_sets_theorem_l2456_245683
