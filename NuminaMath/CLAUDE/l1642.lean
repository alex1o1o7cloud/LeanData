import Mathlib

namespace NUMINAMATH_CALUDE_total_volume_of_boxes_l1642_164213

-- Define the number of boxes
def num_boxes : ℕ := 4

-- Define the edge length of each box in feet
def edge_length : ℝ := 6

-- Define the volume of a single box
def single_box_volume : ℝ := edge_length ^ 3

-- Theorem stating the total volume of all boxes
theorem total_volume_of_boxes : single_box_volume * num_boxes = 864 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_boxes_l1642_164213


namespace NUMINAMATH_CALUDE_even_mono_increasing_negative_l1642_164259

-- Define an even function that is monotonically increasing on [0, +∞)
def EvenMonoIncreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 ≤ x ∧ x < y → f x ≤ f y)

-- Theorem statement
theorem even_mono_increasing_negative (f : ℝ → ℝ) (a b : ℝ) 
  (hf : EvenMonoIncreasing f) (hab : a < b) (hneg : b < 0) : 
  f a > f b := by
  sorry

end NUMINAMATH_CALUDE_even_mono_increasing_negative_l1642_164259


namespace NUMINAMATH_CALUDE_total_people_is_803_l1642_164288

/-- The number of parents in the program -/
def num_parents : ℕ := 105

/-- The number of pupils in the program -/
def num_pupils : ℕ := 698

/-- The total number of people in the program -/
def total_people : ℕ := num_parents + num_pupils

/-- Theorem stating that the total number of people in the program is 803 -/
theorem total_people_is_803 : total_people = 803 := by
  sorry

end NUMINAMATH_CALUDE_total_people_is_803_l1642_164288


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1642_164263

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence property
  a 3 + a 5 = 2 →                                       -- given condition
  a 4 = 1 :=                                            -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1642_164263


namespace NUMINAMATH_CALUDE_technician_round_trip_completion_l1642_164245

theorem technician_round_trip_completion (D : ℝ) (h : D > 0) : 
  let total_distance : ℝ := 2 * D
  let completed_distance : ℝ := D + 0.2 * D
  (completed_distance / total_distance) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_technician_round_trip_completion_l1642_164245


namespace NUMINAMATH_CALUDE_pen_price_payment_l1642_164275

/-- Given the price of a pen and the number of pens bought, determine if the price and total payment are constants or variables -/
theorem pen_price_payment (x : ℕ) (y : ℝ) : 
  (∀ n : ℕ, 3 * n = 3 * n) ∧ (∃ m : ℕ, y ≠ 3 * m) := by
  sorry

end NUMINAMATH_CALUDE_pen_price_payment_l1642_164275


namespace NUMINAMATH_CALUDE_basketball_points_l1642_164296

theorem basketball_points (T : ℕ) : 
  T + (T + 6) + (2 * T + 4) = 26 → T = 4 := by sorry

end NUMINAMATH_CALUDE_basketball_points_l1642_164296


namespace NUMINAMATH_CALUDE_solutions_for_20_l1642_164266

/-- The number of integer solutions for |x| + |y| = n -/
def num_solutions (n : ℕ) : ℕ := 4 * n

/-- Given conditions -/
axiom solution_1 : num_solutions 1 = 4
axiom solution_2 : num_solutions 2 = 8
axiom solution_3 : num_solutions 3 = 12

/-- Theorem: The number of different integer solutions for |x| + |y| = 20 is 80 -/
theorem solutions_for_20 : num_solutions 20 = 80 := by sorry

end NUMINAMATH_CALUDE_solutions_for_20_l1642_164266


namespace NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l1642_164279

-- Define the conditions
def p (x : ℝ) : Prop := abs x > 1
def q (x : ℝ) : Prop := x < -2

-- State the theorem
theorem neg_p_sufficient_not_necessary_for_neg_q :
  (∀ x : ℝ, ¬(p x) → ¬(q x)) ∧ (∃ x : ℝ, ¬(q x) ∧ p x) := by
  sorry

end NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l1642_164279


namespace NUMINAMATH_CALUDE_poster_system_area_l1642_164254

/-- Represents a rectangular poster --/
structure Poster where
  length : ℝ
  width : ℝ

/-- Calculates the area of a poster --/
def poster_area (p : Poster) : ℝ := p.length * p.width

/-- Represents the system of overlapping posters --/
structure PosterSystem where
  posters : List Poster
  num_intersections : ℕ

/-- Theorem: The total area covered by the poster system is 96 square feet --/
theorem poster_system_area (ps : PosterSystem) : 
  ps.posters.length = 4 ∧ 
  (∀ p ∈ ps.posters, p.length = 15 ∧ p.width = 2) ∧
  ps.num_intersections = 3 →
  (ps.posters.map poster_area).sum - ps.num_intersections * 8 = 96 := by
  sorry

#check poster_system_area

end NUMINAMATH_CALUDE_poster_system_area_l1642_164254


namespace NUMINAMATH_CALUDE_baker_cakes_left_l1642_164285

/-- Given a baker who made a total of 217 cakes and sold 145 of them,
    prove that the number of cakes left is 72. -/
theorem baker_cakes_left (total : ℕ) (sold : ℕ) (h1 : total = 217) (h2 : sold = 145) :
  total - sold = 72 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_left_l1642_164285


namespace NUMINAMATH_CALUDE_integers_between_neg_one_third_and_two_l1642_164253

theorem integers_between_neg_one_third_and_two :
  ∀ x : ℤ, -1/3 < (x : ℚ) ∧ (x : ℚ) < 2 → x = 0 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_integers_between_neg_one_third_and_two_l1642_164253


namespace NUMINAMATH_CALUDE_common_chord_circle_center_on_line_smallest_circle_l1642_164201

-- Define the two circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0

-- Define points A and B as the intersection of C₁ and C₂
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 2)

-- Theorem for the common chord
theorem common_chord : 
  ∀ x y : ℝ, C₁ x y ∧ C₂ x y → x - 2*y + 4 = 0 :=
by sorry

-- Theorem for the circle with center on y = -x
theorem circle_center_on_line : 
  ∃ h k : ℝ, h = -k ∧ 
  (A.1 - h)^2 + (A.2 - k)^2 = (B.1 - h)^2 + (B.2 - k)^2 ∧
  ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = 10 ↔ 
  ((x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) :=
by sorry

-- Theorem for the smallest circle
theorem smallest_circle : 
  ∀ x y : ℝ, (x + 2)^2 + (y - 1)^2 = 5 ↔ 
  ((x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_circle_center_on_line_smallest_circle_l1642_164201


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l1642_164298

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (u v : ℝ × ℝ) : ℝ :=
  |u.1 * v.2 - u.2 * v.1|

theorem parallelogram_area_example : 
  let u : ℝ × ℝ := (4, 7)
  let z : ℝ × ℝ := (-6, 3)
  parallelogramArea u z = 54 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l1642_164298


namespace NUMINAMATH_CALUDE_parallel_line_m_value_l1642_164250

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Get the line passing through two points -/
def line_through_points (p1 p2 : Point) : Line :=
  { a := p2.y - p1.y
    b := p1.x - p2.x
    c := p2.x * p1.y - p1.x * p2.y }

/-- The main theorem -/
theorem parallel_line_m_value :
  ∀ m : ℝ,
  let A : Point := ⟨-2, m⟩
  let B : Point := ⟨m, 4⟩
  let L1 : Line := line_through_points A B
  let L2 : Line := ⟨2, 1, -1⟩
  are_parallel L1 L2 → m = -8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_m_value_l1642_164250


namespace NUMINAMATH_CALUDE_sequence_constant_condition_general_term_l1642_164294

/-- Fibonacci sequence -/
def fib : ℕ → ℤ
| 0 => 1
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- The sequence a_n -/
noncomputable def a (x y : ℝ) : ℕ → ℝ
| 0 => x
| 1 => y
| (n + 2) => (a x y (n + 1) * a x y n + 1) / (a x y (n + 1) + a x y n)

theorem sequence_constant_condition (x y : ℝ) :
  (∃ n₀ : ℕ, ∀ n ≥ n₀, a x y (n + 1) = a x y n) ↔ (abs x = 1 ∧ y ≠ -x) :=
sorry

theorem general_term (x y : ℝ) (n : ℕ) :
  a x y n = ((x + 1)^(fib (n - 2)) * (y + 1)^(fib (n - 1)) + (x + 1)^(fib (n - 2)) * (y - 1)^(fib (n - 1))) /
            ((x + 1)^(fib (n - 2)) * (y + 1)^(fib (n - 1)) - (x - 1)^(fib (n - 2)) * (y - 1)^(fib (n - 1))) :=
sorry

end NUMINAMATH_CALUDE_sequence_constant_condition_general_term_l1642_164294


namespace NUMINAMATH_CALUDE_rental_distance_theorem_l1642_164295

/-- Calculates the distance driven given rental parameters and total cost -/
def distance_driven (daily_rate : ℚ) (mile_rate : ℚ) (total_cost : ℚ) : ℚ :=
  (total_cost - daily_rate) / mile_rate

theorem rental_distance_theorem (daily_rate mile_rate total_cost : ℚ) :
  daily_rate = 29 →
  mile_rate = 0.08 →
  total_cost = 46.12 →
  distance_driven daily_rate mile_rate total_cost = 214 := by
  sorry

end NUMINAMATH_CALUDE_rental_distance_theorem_l1642_164295


namespace NUMINAMATH_CALUDE_profit_share_difference_l1642_164277

/-- Represents the initial capital and interest rate for each partner --/
structure Partner where
  capital : ℕ
  rate : ℚ

/-- Calculates the interest earned by a partner --/
def interest (p : Partner) : ℚ := p.capital * p.rate

/-- Calculates the profit share of a partner --/
def profitShare (p : Partner) (totalProfit : ℕ) : ℚ :=
  p.capital + interest p

theorem profit_share_difference
  (a b c : Partner)
  (ha : a.capital = 8000 ∧ a.rate = 5/100)
  (hb : b.capital = 10000 ∧ b.rate = 6/100)
  (hc : c.capital = 12000 ∧ c.rate = 7/100)
  (totalProfit : ℕ)
  (hProfit : profitShare b totalProfit = 13600) :
  profitShare c totalProfit - profitShare a totalProfit = 4440 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_difference_l1642_164277


namespace NUMINAMATH_CALUDE_soda_price_calculation_l1642_164242

def initial_amount : ℕ := 500
def rice_packets : ℕ := 2
def rice_price : ℕ := 20
def wheat_packets : ℕ := 3
def wheat_price : ℕ := 25
def remaining_balance : ℕ := 235

theorem soda_price_calculation :
  ∃ (soda_price : ℕ),
    initial_amount - (rice_packets * rice_price + wheat_packets * wheat_price + soda_price) = remaining_balance ∧
    soda_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_calculation_l1642_164242


namespace NUMINAMATH_CALUDE_arithmetic_sequence_b_l1642_164240

def arithmetic_sequence (a₁ a₂ a₃ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ = a₁ + d ∧ a₃ = a₂ + d

theorem arithmetic_sequence_b (b : ℝ) 
  (h₁ : arithmetic_sequence 120 b (1/5))
  (h₂ : b > 0) : 
  b = 60.1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_b_l1642_164240


namespace NUMINAMATH_CALUDE_petya_wins_l1642_164289

/-- Represents a position on the board -/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents the game state -/
structure GameState :=
  (board : Fin 101 → Fin 101 → Bool)
  (lastMoveLength : Nat)

/-- Represents a move in the game -/
inductive Move
  | Initial : Position → Move
  | Strip : Position → Nat → Move

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  match move with
  | Move.Initial _ => state.lastMoveLength = 0
  | Move.Strip _ n => n = state.lastMoveLength ∨ n = state.lastMoveLength + 1

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if a player has a winning strategy from a given state -/
def hasWinningStrategy (state : GameState) (isFirstPlayer : Bool) : Prop :=
  sorry

/-- The main theorem stating that the first player (Petya) has a winning strategy -/
theorem petya_wins :
  ∃ (initialMove : Move),
    isValidMove { board := λ _ _ => false, lastMoveLength := 0 } initialMove ∧
    hasWinningStrategy (applyMove { board := λ _ _ => false, lastMoveLength := 0 } initialMove) true :=
  sorry

end NUMINAMATH_CALUDE_petya_wins_l1642_164289


namespace NUMINAMATH_CALUDE_grid_division_equal_areas_l1642_164207

-- Define the grid
def grid_size : ℕ := 6

-- Define point P
def P : ℚ × ℚ := (3, 3)

-- Define points J and T
def J : ℚ × ℚ := (0, 4)
def T : ℚ × ℚ := (6, 4)

-- Function to calculate area of a triangle
def triangle_area (a b c : ℚ × ℚ) : ℚ :=
  let (x1, y1) := a
  let (x2, y2) := b
  let (x3, y3) := c
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) : ℚ)

-- Theorem statement
theorem grid_division_equal_areas :
  let area1 := triangle_area P J (0, 0)
  let area2 := triangle_area P T (grid_size, 0)
  let area3 := (grid_size * grid_size : ℚ) - area1 - area2
  area1 = area2 ∧ area2 = area3 := by sorry

end NUMINAMATH_CALUDE_grid_division_equal_areas_l1642_164207


namespace NUMINAMATH_CALUDE_jellybean_problem_l1642_164287

theorem jellybean_problem :
  ∃ n : ℕ, n ≥ 200 ∧ n % 17 = 15 ∧ ∀ m : ℕ, m ≥ 200 ∧ m % 17 = 15 → m ≥ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l1642_164287


namespace NUMINAMATH_CALUDE_f_zero_at_one_f_zero_at_five_f_value_at_three_l1642_164273

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

/-- The function f has a zero at x = 1 -/
theorem f_zero_at_one : f 1 = 0 := by sorry

/-- The function f has a zero at x = 5 -/
theorem f_zero_at_five : f 5 = 0 := by sorry

/-- The function f takes the value 8 when x = 3 -/
theorem f_value_at_three : f 3 = 8 := by sorry

end NUMINAMATH_CALUDE_f_zero_at_one_f_zero_at_five_f_value_at_three_l1642_164273


namespace NUMINAMATH_CALUDE_trader_profit_equation_l1642_164220

/-- The trader's profit after a week of sales -/
def trader_profit : ℝ := 960

/-- The amount of donations received -/
def donations : ℝ := 310

/-- The trader's goal amount -/
def goal : ℝ := 610

/-- The amount above the goal -/
def above_goal : ℝ := 180

theorem trader_profit_equation :
  trader_profit / 2 + donations = goal + above_goal :=
by sorry

end NUMINAMATH_CALUDE_trader_profit_equation_l1642_164220


namespace NUMINAMATH_CALUDE_pushup_sets_l1642_164291

theorem pushup_sets (total_pushups : ℕ) (sets : ℕ) (reduction : ℕ) : 
  total_pushups = 40 → sets = 3 → reduction = 5 → 
  ∃ x : ℕ, x + x + (x - reduction) = total_pushups ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_pushup_sets_l1642_164291


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l1642_164251

/-- The standard form of a hyperbola with center at the origin --/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point (x, y) is on the hyperbola --/
def Hyperbola.contains (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Check if two hyperbolas have the same asymptotes --/
def same_asymptotes (h1 h2 : Hyperbola) : Prop :=
  h1.a^2 / h1.b^2 = h2.a^2 / h2.b^2

theorem hyperbola_theorem (h1 h2 : Hyperbola) :
  h1.a^2 = 3 ∧ h1.b^2 = 12 ∧
  h2.a^2 = 1 ∧ h2.b^2 = 4 →
  same_asymptotes h1 h2 ∧ h1.contains 2 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l1642_164251


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1642_164264

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (· + 1)

/-- Theorem: In a geometric sequence where the fourth term is 6! and the seventh term is 7!, the first term is 720/7. -/
theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geometric : IsGeometricSequence a)
  (h_fourth : a 4 = factorial 6)
  (h_seventh : a 7 = factorial 7) :
  a 1 = 720 / 7 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1642_164264


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l1642_164237

theorem absolute_value_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ + 3| = 15) ∧ 
  (|x₂ + 3| = 15) ∧ 
  (x₁ ≠ x₂) ∧ 
  (x₁ - x₂ = 30 ∨ x₂ - x₁ = 30) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l1642_164237


namespace NUMINAMATH_CALUDE_distribution_count_l1642_164221

theorem distribution_count (num_items : ℕ) (num_recipients : ℕ) : 
  num_items = 6 → num_recipients = 8 → num_recipients ^ num_items = 262144 := by
  sorry

end NUMINAMATH_CALUDE_distribution_count_l1642_164221


namespace NUMINAMATH_CALUDE_price_per_square_foot_l1642_164258

def house_area : ℝ := 2400
def barn_area : ℝ := 1000
def total_property_value : ℝ := 333200

theorem price_per_square_foot :
  total_property_value / (house_area + barn_area) = 98 := by
sorry

end NUMINAMATH_CALUDE_price_per_square_foot_l1642_164258


namespace NUMINAMATH_CALUDE_expression_evaluation_l1642_164256

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (hsum : x + y ≠ 0) (hsum_sq : x^2 + y^2 ≠ 0) :
  (x^2 + y^2)⁻¹ * ((x + y)⁻¹ + (x / y)⁻¹) = (1 + y) / ((x^2 + y^2) * (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1642_164256


namespace NUMINAMATH_CALUDE_max_value_abcd_l1642_164224

theorem max_value_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a * b * c * d * (a + b + c + d)) / ((a + b)^2 * (c + d)^2) ≤ (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_abcd_l1642_164224


namespace NUMINAMATH_CALUDE_modulo_graph_intercepts_sum_l1642_164236

theorem modulo_graph_intercepts_sum (m : Nat) (x₀ y₀ : Nat) : m = 7 →
  0 ≤ x₀ → x₀ < m →
  0 ≤ y₀ → y₀ < m →
  (2 * x₀) % m = 1 % m →
  (3 * y₀ + 1) % m = 0 →
  x₀ + y₀ = 6 := by
sorry

end NUMINAMATH_CALUDE_modulo_graph_intercepts_sum_l1642_164236


namespace NUMINAMATH_CALUDE_sequence_properties_l1642_164268

def S (n : ℕ) : ℤ := -n^2 + 7*n

def a (n : ℕ) : ℤ := S n - S (n-1)

theorem sequence_properties :
  (∀ n : ℕ, a n = -2*n + 8) ∧
  (∀ n : ℕ, n > 4 → a n < 0) ∧
  (∀ n : ℕ, S n ≤ S 3 ∧ S n ≤ S 4) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1642_164268


namespace NUMINAMATH_CALUDE_min_value_inequality_l1642_164261

theorem min_value_inequality (a : ℝ) : 
  (∀ x y : ℝ, |x| + |y| ≤ 1 → |2*x - 3*y + 3/2| + |y - 1| + |2*y - x - 3| ≤ a) ↔ 
  23/2 ≤ a :=
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1642_164261


namespace NUMINAMATH_CALUDE_temperature_conversion_l1642_164217

theorem temperature_conversion (C F : ℝ) : 
  C = 35 → C = (4/7) * (F - 40) → F = 101.25 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l1642_164217


namespace NUMINAMATH_CALUDE_arithmetic_harmonic_means_equal_implies_equal_values_l1642_164238

theorem arithmetic_harmonic_means_equal_implies_equal_values (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 2) 
  (h_harmonic : 2 / (1/a + 1/b) = 2) : 
  a = 2 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_harmonic_means_equal_implies_equal_values_l1642_164238


namespace NUMINAMATH_CALUDE_integer_pair_sum_l1642_164281

theorem integer_pair_sum (m n : ℤ) (h : (m^2 + m*n + n^2) / (m + 2*n) = 13/3) : 
  m + 2*n = 9 := by
sorry

end NUMINAMATH_CALUDE_integer_pair_sum_l1642_164281


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_18_27_45_l1642_164208

def arithmetic_mean (a b c : ℕ) : ℚ :=
  (a + b + c : ℚ) / 3

theorem arithmetic_mean_of_18_27_45 :
  arithmetic_mean 18 27 45 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_18_27_45_l1642_164208


namespace NUMINAMATH_CALUDE_CQRP_is_parallelogram_l1642_164278

-- Define the triangle ABC
structure Triangle (α β : ℝ) :=
  (A B C : ℂ)
  (angle_condition : α > 45 ∧ β > 45)

-- Define the construction of points R, P, and Q
def construct_R (t : Triangle α β) : ℂ :=
  t.B + (t.A - t.B) * Complex.I

def construct_P (t : Triangle α β) : ℂ :=
  t.C + (t.B - t.C) * (-Complex.I)

def construct_Q (t : Triangle α β) : ℂ :=
  t.C + (t.A - t.C) * Complex.I

-- State the theorem
theorem CQRP_is_parallelogram (α β : ℝ) (t : Triangle α β) :
  let R := construct_R t
  let P := construct_P t
  let Q := construct_Q t
  (R + P) / 2 = (t.C + Q) / 2 := by sorry

end NUMINAMATH_CALUDE_CQRP_is_parallelogram_l1642_164278


namespace NUMINAMATH_CALUDE_city_death_rate_l1642_164267

/-- Represents the population dynamics of a city --/
structure CityPopulation where
  birth_rate : ℕ  -- Birth rate per two seconds
  net_increase : ℕ  -- Net population increase per day

/-- Calculates the death rate per two seconds given city population data --/
def death_rate (city : CityPopulation) : ℕ :=
  let seconds_per_day : ℕ := 24 * 60 * 60
  let birth_rate_per_second : ℕ := city.birth_rate / 2
  let net_increase_per_second : ℕ := city.net_increase / seconds_per_day
  2 * (birth_rate_per_second - net_increase_per_second)

/-- Theorem stating that for the given city data, the death rate is 6 people every two seconds --/
theorem city_death_rate :
  let city : CityPopulation := { birth_rate := 8, net_increase := 86400 }
  death_rate city = 6 := by
  sorry

end NUMINAMATH_CALUDE_city_death_rate_l1642_164267


namespace NUMINAMATH_CALUDE_negation_of_forall_quadratic_inequality_l1642_164276

theorem negation_of_forall_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_quadratic_inequality_l1642_164276


namespace NUMINAMATH_CALUDE_second_largest_divided_by_smallest_remainder_l1642_164262

theorem second_largest_divided_by_smallest_remainder : ∃ (a b c d : ℕ),
  (a = 10 ∧ b = 11 ∧ c = 12 ∧ d = 13) →
  (a < b ∧ b < c ∧ c < d) →
  c % a = 2 := by
sorry

end NUMINAMATH_CALUDE_second_largest_divided_by_smallest_remainder_l1642_164262


namespace NUMINAMATH_CALUDE_log_equation_solution_l1642_164292

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log (x^3) / Real.log 9 = 9 →
  x = 3^(18/5) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1642_164292


namespace NUMINAMATH_CALUDE_white_tshirts_per_pack_is_five_l1642_164243

/-- The number of white T-shirts in one pack -/
def white_tshirts_per_pack : ℕ := sorry

/-- The number of packs of white T-shirts bought -/
def white_packs : ℕ := 2

/-- The number of packs of blue T-shirts bought -/
def blue_packs : ℕ := 4

/-- The number of blue T-shirts in one pack -/
def blue_tshirts_per_pack : ℕ := 3

/-- The cost of one T-shirt in dollars -/
def cost_per_tshirt : ℕ := 3

/-- The total cost of all T-shirts in dollars -/
def total_cost : ℕ := 66

theorem white_tshirts_per_pack_is_five :
  white_tshirts_per_pack = 5 :=
by
  sorry

#check white_tshirts_per_pack_is_five

end NUMINAMATH_CALUDE_white_tshirts_per_pack_is_five_l1642_164243


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_product_l1642_164246

theorem arithmetic_geometric_mean_product : 
  ∀ a b : ℝ, 
  (a = (1 + 2) / 2) → 
  (b^2 = (-1) * (-16)) → 
  (a * b = 6 ∨ a * b = -6) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_product_l1642_164246


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1642_164271

theorem binomial_coefficient_ratio (n : ℕ) : 
  (2^3 * (n.choose 3) = 4 * 2^2 * (n.choose 2)) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1642_164271


namespace NUMINAMATH_CALUDE_busy_squirrel_nuts_calculation_l1642_164210

/-- The number of nuts stockpiled per day by each busy squirrel -/
def busy_squirrel_nuts_per_day : ℕ := 30

/-- The number of busy squirrels -/
def num_busy_squirrels : ℕ := 2

/-- The number of nuts stockpiled per day by the sleepy squirrel -/
def sleepy_squirrel_nuts_per_day : ℕ := 20

/-- The number of days the squirrels have been stockpiling -/
def num_days : ℕ := 40

/-- The total number of nuts in Mason's car -/
def total_nuts : ℕ := 3200

theorem busy_squirrel_nuts_calculation :
  num_busy_squirrels * busy_squirrel_nuts_per_day * num_days + 
  sleepy_squirrel_nuts_per_day * num_days = total_nuts :=
by sorry

end NUMINAMATH_CALUDE_busy_squirrel_nuts_calculation_l1642_164210


namespace NUMINAMATH_CALUDE_morning_trip_fare_correct_afternoon_trip_fare_formula_afternoon_trip_fare_specific_l1642_164248

/- Define the time periods and corresponding rates -/
def normal_mileage_rate : ℝ := 2.20
def early_morning_mileage_rate : ℝ := 2.80
def peak_mileage_rate : ℝ := 2.75
def normal_time_rate : ℝ := 0.38
def peak_time_rate : ℝ := 0.47

/- Define the fare calculation function -/
def calculate_fare (distance : ℝ) (time : ℝ) (mileage_rate : ℝ) (time_rate : ℝ) : ℝ :=
  distance * mileage_rate + time * time_rate

/- Theorem for the morning trip -/
theorem morning_trip_fare_correct :
  calculate_fare 6 10 early_morning_mileage_rate normal_time_rate = 20.6 := by sorry

/- Theorem for the afternoon trip (general formula) -/
theorem afternoon_trip_fare_formula (x : ℝ) (h : x ≤ 30) :
  calculate_fare x (x / 30 * 60) peak_mileage_rate peak_time_rate = 3.69 * x := by sorry

/- Theorem for the afternoon trip when x = 8 -/
theorem afternoon_trip_fare_specific :
  calculate_fare 8 16 peak_mileage_rate peak_time_rate = 29.52 := by sorry

end NUMINAMATH_CALUDE_morning_trip_fare_correct_afternoon_trip_fare_formula_afternoon_trip_fare_specific_l1642_164248


namespace NUMINAMATH_CALUDE_base_equality_l1642_164244

/-- Converts a base 6 number to its decimal equivalent -/
def base6ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number in base b to its decimal equivalent -/
def baseBToDecimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- The unique positive integer b that satisfies 34₆ = 121ᵦ is 3 -/
theorem base_equality : ∃! (b : ℕ), b > 0 ∧ base6ToDecimal 34 = baseBToDecimal 121 b ∧ b = 3 := by sorry

end NUMINAMATH_CALUDE_base_equality_l1642_164244


namespace NUMINAMATH_CALUDE_team_leaders_problem_l1642_164290

theorem team_leaders_problem (m n : ℕ) :
  (10 ≥ m ∧ m > n ∧ n ≥ 4) →
  (Nat.choose (m + n) 2 * (Nat.choose m 2 + Nat.choose n 2) = 
   Nat.choose (m + n) 2 * (m * n)) →
  (m = 10 ∧ n = 6) :=
by sorry

end NUMINAMATH_CALUDE_team_leaders_problem_l1642_164290


namespace NUMINAMATH_CALUDE_percentage_problem_l1642_164223

theorem percentage_problem : ∃ X : ℝ, 
  (X / 100 * 100 = (0.6 * 80 + 22)) ∧ 
  (X = 70) := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l1642_164223


namespace NUMINAMATH_CALUDE_width_to_length_ratio_l1642_164286

/-- A rectangle represents a rectangular hall --/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Properties of the rectangular hall --/
def RectangleProperties (r : Rectangle) : Prop :=
  r.width > 0 ∧ 
  r.length > 0 ∧ 
  r.width * r.length = 450 ∧ 
  r.length - r.width = 15

/-- Theorem stating the ratio of width to length --/
theorem width_to_length_ratio (r : Rectangle) 
  (h : RectangleProperties r) : r.width / r.length = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_width_to_length_ratio_l1642_164286


namespace NUMINAMATH_CALUDE_total_pages_theorem_l1642_164218

/-- The number of pages Jairus read -/
def jairus_pages : ℕ := 20

/-- The number of pages Arniel read -/
def arniel_pages : ℕ := 2 * jairus_pages + 2

/-- The total number of pages read by Jairus and Arniel -/
def total_pages : ℕ := jairus_pages + arniel_pages

theorem total_pages_theorem : total_pages = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_theorem_l1642_164218


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l1642_164249

theorem smallest_number_with_conditions : ∃! n : ℕ, 
  (n % 11 = 0) ∧
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 8 → n % k = 1) ∧
  (∀ m : ℕ, 
    (m % 11 = 0) ∧ 
    (∀ k : ℕ, 2 ≤ k ∧ k ≤ 8 → m % k = 1) → 
    n ≤ m) ∧
  n = 6721 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l1642_164249


namespace NUMINAMATH_CALUDE_wall_painting_fraction_l1642_164202

theorem wall_painting_fraction (total_time minutes : ℕ) (fraction : ℚ) : 
  total_time = 60 → 
  minutes = 12 → 
  fraction = minutes / total_time → 
  fraction = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_wall_painting_fraction_l1642_164202


namespace NUMINAMATH_CALUDE_charity_fundraising_l1642_164225

theorem charity_fundraising (total : ℕ) (people : ℕ) (raised : ℕ) (target : ℕ) :
  total = 2100 →
  people = 8 →
  raised = 150 →
  target = 279 →
  (total - raised) / (people - 1) = target := by
  sorry

end NUMINAMATH_CALUDE_charity_fundraising_l1642_164225


namespace NUMINAMATH_CALUDE_amc10_paths_l1642_164214

/-- The number of 'M's adjacent to the central 'A' -/
def num_m_adj_a : ℕ := 4

/-- The number of 'C's adjacent to each 'M' -/
def num_c_adj_m : ℕ := 4

/-- The number of '10's adjacent to each 'C' -/
def num_10_adj_c : ℕ := 5

/-- The total number of paths to spell "AMC10" -/
def total_paths : ℕ := num_m_adj_a * num_c_adj_m * num_10_adj_c

theorem amc10_paths : total_paths = 80 := by
  sorry

end NUMINAMATH_CALUDE_amc10_paths_l1642_164214


namespace NUMINAMATH_CALUDE_count_of_satisfying_integers_l1642_164272

/-- The number of integers satisfying the equation -/
def solution_count : ℕ := 40200

/-- The equation to be satisfied -/
def satisfies_equation (n : ℤ) : Prop :=
  1 + ⌊(200 * n) / 201⌋ = ⌈(198 * n) / 200⌉

theorem count_of_satisfying_integers :
  (∃! (s : Finset ℤ), s.card = solution_count ∧ ∀ n, n ∈ s ↔ satisfies_equation n) :=
sorry

end NUMINAMATH_CALUDE_count_of_satisfying_integers_l1642_164272


namespace NUMINAMATH_CALUDE_angle_D_value_l1642_164293

theorem angle_D_value (A B C D : ℝ) 
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 40)
  (h4 : B + C = 130) :
  D = 40 := by sorry

end NUMINAMATH_CALUDE_angle_D_value_l1642_164293


namespace NUMINAMATH_CALUDE_midpoint_movement_l1642_164206

/-- Given two points A and B in a Cartesian plane, their midpoint, and their new positions after
    movement, prove the new midpoint and its distance from the original midpoint. -/
theorem midpoint_movement (a b c d m n : ℝ) :
  let M : ℝ × ℝ := (m, n)
  let A : ℝ × ℝ := (a, b)
  let B : ℝ × ℝ := (c, d)
  let A' : ℝ × ℝ := (a + 3, b + 5)
  let B' : ℝ × ℝ := (c - 6, d - 3)
  let M' : ℝ × ℝ := ((a + 3 + c - 6) / 2, (b + 5 + d - 3) / 2)
  (M = ((a + c) / 2, (b + d) / 2)) →
  (M' = (m - 3 / 2, n + 1) ∧
   Real.sqrt ((m - 3 / 2 - m) ^ 2 + (n + 1 - n) ^ 2) = Real.sqrt 13 / 2) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_movement_l1642_164206


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l1642_164255

theorem complex_product_pure_imaginary (a : ℝ) : 
  (Complex.I + 1) * (Complex.I * a + 1) = Complex.I * (Complex.I.im * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l1642_164255


namespace NUMINAMATH_CALUDE_integer_solutions_exist_l1642_164230

theorem integer_solutions_exist : ∃ (k x : ℤ), (k - 5) * x + 6 = 1 - 5 * x ∧
  ((k = 1 ∧ x = -5) ∨ (k = -1 ∧ x = 5) ∨ (k = 5 ∧ x = -1) ∨ (k = -5 ∧ x = 1)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_exist_l1642_164230


namespace NUMINAMATH_CALUDE_reinforcement_size_l1642_164231

/-- Calculates the size of reinforcement given initial garrison size, initial provision duration,
    days before reinforcement, and remaining provision duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_provisions : ℕ) 
                            (days_before_reinforcement : ℕ) (remaining_provisions : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_provisions
  let used_provisions := initial_garrison * days_before_reinforcement
  let remaining_total := total_provisions - used_provisions
  (remaining_total / remaining_provisions) - initial_garrison

theorem reinforcement_size :
  let initial_garrison : ℕ := 2000
  let initial_provisions : ℕ := 54
  let days_before_reinforcement : ℕ := 21
  let remaining_provisions : ℕ := 20
  calculate_reinforcement initial_garrison initial_provisions days_before_reinforcement remaining_provisions = 1300 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_l1642_164231


namespace NUMINAMATH_CALUDE_quadratic_root_property_l1642_164228

theorem quadratic_root_property : ∀ a b : ℝ, 
  (a^2 - 3*a + 1 = 0) → (b^2 - 3*b + 1 = 0) → (a + b - a*b = 2) := by sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l1642_164228


namespace NUMINAMATH_CALUDE_empty_solution_set_iff_k_ge_one_l1642_164257

-- Define the inequality function
def f (k x : ℝ) : ℝ := k * x^2 - 2 * abs (x - 1) + 3 * k

-- Define the property of having an empty solution set
def has_empty_solution_set (k : ℝ) : Prop :=
  ∀ x, f k x ≥ 0

-- State the theorem
theorem empty_solution_set_iff_k_ge_one :
  ∀ k, has_empty_solution_set k ↔ k ≥ 1 := by sorry

end NUMINAMATH_CALUDE_empty_solution_set_iff_k_ge_one_l1642_164257


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1642_164299

/-- A quadratic function with vertex (3, 2) passing through (-2, -43) has a = -1.8 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x, (a * x^2 + b * x + c) = a * (x - 3)^2 + 2) → 
  (a * (-2)^2 + b * (-2) + c = -43) → 
  a = -1.8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1642_164299


namespace NUMINAMATH_CALUDE_cos_ninety_degrees_l1642_164260

theorem cos_ninety_degrees : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_ninety_degrees_l1642_164260


namespace NUMINAMATH_CALUDE_min_ratio_of_intersections_l1642_164219

theorem min_ratio_of_intersections (a : ℝ) (ha : a > 0) :
  let f (x : ℝ) := |Real.log x / Real.log 4|
  let x_A := (4 : ℝ) ^ (-a)
  let x_B := (4 : ℝ) ^ a
  let x_C := (4 : ℝ) ^ (-18 / (2 * a + 1))
  let x_D := (4 : ℝ) ^ (18 / (2 * a + 1))
  let m := |x_A - x_C|
  let n := |x_B - x_D|
  ∃ (a_min : ℝ), ∀ (a : ℝ), a > 0 → n / m ≥ 2^11 ∧ n / m = 2^11 ↔ a = a_min :=
sorry

end NUMINAMATH_CALUDE_min_ratio_of_intersections_l1642_164219


namespace NUMINAMATH_CALUDE_simple_interest_time_period_l1642_164227

theorem simple_interest_time_period 
  (P : ℝ) -- Principal sum
  (R : ℝ) -- Rate of interest per annum
  (T : ℝ) -- Time period in years
  (h1 : R = 4) -- Given rate of interest is 4%
  (h2 : P / 5 = (P * R * T) / 100) -- Simple interest is one-fifth of principal and follows the formula
  : T = 5 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_time_period_l1642_164227


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_intersection_l1642_164269

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Finds the intersection point of two line segments -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem quadrilateral_diagonal_intersection 
  (q : Quadrilateral) 
  (hConvex : isConvex q) 
  (hAB : distance q.A q.B = 12)
  (hCD : distance q.C q.D = 15)
  (hAC : distance q.A q.C = 18)
  (E : Point)
  (hE : E = lineIntersection q.A q.C q.B q.D)
  (hAreas : triangleArea q.A E q.D = triangleArea q.B E q.C) :
  distance q.A E = 8 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_intersection_l1642_164269


namespace NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l1642_164233

theorem square_sum_given_diff_and_product (x y : ℝ) 
  (h1 : x - y = 12) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 162 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l1642_164233


namespace NUMINAMATH_CALUDE_division_problem_l1642_164247

theorem division_problem (n : ℕ) : 
  (n / 6 = 8) → (n % 6 ≤ 5) → (n % 6 = 5) → (n = 53) :=
by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1642_164247


namespace NUMINAMATH_CALUDE_max_sum_reciprocal_cubes_l1642_164280

/-- The roots of a cubic polynomial satisfying a specific condition -/
structure CubicRoots where
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ
  sum_eq_sum_squares : r₁ + r₂ + r₃ = r₁^2 + r₂^2 + r₃^2

/-- The coefficients of a cubic polynomial -/
structure CubicCoeffs where
  s : ℝ
  p : ℝ
  q : ℝ

/-- The theorem stating the maximum value of the sum of reciprocal cubes of roots -/
theorem max_sum_reciprocal_cubes (roots : CubicRoots) (coeffs : CubicCoeffs) 
  (vieta₁ : roots.r₁ + roots.r₂ + roots.r₃ = coeffs.s)
  (vieta₂ : roots.r₁ * roots.r₂ + roots.r₂ * roots.r₃ + roots.r₃ * roots.r₁ = coeffs.p)
  (vieta₃ : roots.r₁ * roots.r₂ * roots.r₃ = coeffs.q) :
  ∃ (max : ℝ), max = 3 ∧ ∀ (roots' : CubicRoots),
    (1 / roots'.r₁^3 + 1 / roots'.r₂^3 + 1 / roots'.r₃^3) ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_sum_reciprocal_cubes_l1642_164280


namespace NUMINAMATH_CALUDE_exponent_rule_product_power_l1642_164211

theorem exponent_rule_product_power (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_rule_product_power_l1642_164211


namespace NUMINAMATH_CALUDE_triangle_inequality_bound_l1642_164203

theorem triangle_inequality_bound (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + c^2) / (b + c)^2 ≤ (1 : ℝ) / 2 ∧
  ∀ ε > 0, ∃ a' b' c' : ℝ, 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧
    (a'^2 + c'^2) / (b' + c')^2 > (1 : ℝ) / 2 - ε :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_bound_l1642_164203


namespace NUMINAMATH_CALUDE_police_officer_arrangements_l1642_164204

def num_officers : ℕ := 5
def num_intersections : ℕ := 3

def valid_distribution (d : List ℕ) : Prop :=
  d.length = num_intersections ∧
  d.sum = num_officers ∧
  ∀ x ∈ d, 1 ≤ x ∧ x ≤ 3

def arrangements (d : List ℕ) : ℕ := sorry

def arrangements_with_AB_separate : ℕ := sorry

theorem police_officer_arrangements :
  arrangements_with_AB_separate = 114 := by sorry

end NUMINAMATH_CALUDE_police_officer_arrangements_l1642_164204


namespace NUMINAMATH_CALUDE_compound_inequality_solution_l1642_164215

theorem compound_inequality_solution (x : ℝ) :
  (3 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9 * x - 6) ↔ 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_compound_inequality_solution_l1642_164215


namespace NUMINAMATH_CALUDE_factorization_sum_l1642_164252

theorem factorization_sum (a b : ℤ) :
  (∀ x, 25 * x^2 - 155 * x - 150 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = 27 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l1642_164252


namespace NUMINAMATH_CALUDE_one_eighth_of_2_40_l1642_164212

theorem one_eighth_of_2_40 (x : ℕ) : (1 / 8 : ℝ) * 2^40 = 2^x → x = 37 := by
  sorry

end NUMINAMATH_CALUDE_one_eighth_of_2_40_l1642_164212


namespace NUMINAMATH_CALUDE_jerry_throwing_points_l1642_164239

/-- Represents the point system in Mrs. Carlton's class -/
structure PointSystem where
  interrupt_points : ℕ
  insult_points : ℕ
  office_threshold : ℕ

/-- Represents Jerry's behavior -/
structure JerryBehavior where
  interrupts : ℕ
  insults : ℕ
  throws : ℕ

/-- Calculates the points Jerry has accumulated so far -/
def accumulated_points (ps : PointSystem) (jb : JerryBehavior) : ℕ :=
  ps.interrupt_points * jb.interrupts + ps.insult_points * jb.insults

/-- Theorem stating that Jerry gets 25 points for throwing things -/
theorem jerry_throwing_points (ps : PointSystem) (jb : JerryBehavior) :
    ps.interrupt_points = 5 →
    ps.insult_points = 10 →
    ps.office_threshold = 100 →
    jb.interrupts = 2 →
    jb.insults = 4 →
    jb.throws = 2 →
    (ps.office_threshold - accumulated_points ps jb) / jb.throws = 25 := by
  sorry

end NUMINAMATH_CALUDE_jerry_throwing_points_l1642_164239


namespace NUMINAMATH_CALUDE_book_sales_properties_l1642_164229

/-- Represents the daily sales and profit functions for a book selling business -/
structure BookSales where
  cost : ℝ              -- Cost price per book
  min_price : ℝ         -- Minimum selling price
  max_profit_rate : ℝ   -- Maximum profit rate
  base_sales : ℝ        -- Base sales at minimum price
  sales_decrease : ℝ    -- Sales decrease per unit price increase

variable (bs : BookSales)

/-- Daily sales as a function of price -/
def daily_sales (x : ℝ) : ℝ := bs.base_sales - bs.sales_decrease * (x - bs.min_price)

/-- Daily profit as a function of price -/
def daily_profit (x : ℝ) : ℝ := (x - bs.cost) * (daily_sales bs x)

/-- Theorem stating the properties of the book selling business -/
theorem book_sales_properties (bs : BookSales) 
  (h_cost : bs.cost = 40)
  (h_min_price : bs.min_price = 45)
  (h_max_profit_rate : bs.max_profit_rate = 0.5)
  (h_base_sales : bs.base_sales = 310)
  (h_sales_decrease : bs.sales_decrease = 10) :
  -- 1. Daily sales function
  (∀ x, daily_sales bs x = -10 * x + 760) ∧
  -- 2. Selling price range
  (∀ x, bs.min_price ≤ x ∧ x ≤ bs.cost * (1 + bs.max_profit_rate)) ∧
  -- 3. Profit-maximizing price
  (∃ x_max, ∀ x, daily_profit bs x ≤ daily_profit bs x_max ∧ x_max = 58) ∧
  -- 4. Maximum daily profit
  (∃ max_profit, max_profit = daily_profit bs 58 ∧ max_profit = 3240) ∧
  -- 5. Price for $2600 profit
  (∃ x_2600, daily_profit bs x_2600 = 2600 ∧ x_2600 = 50) := by
  sorry

end NUMINAMATH_CALUDE_book_sales_properties_l1642_164229


namespace NUMINAMATH_CALUDE_ratio_change_proof_l1642_164284

theorem ratio_change_proof (x y a : ℚ) : 
  y = 40 →
  x / y = 3 / 4 →
  (x + a) / (y + a) = 4 / 5 →
  a = 10 := by
sorry

end NUMINAMATH_CALUDE_ratio_change_proof_l1642_164284


namespace NUMINAMATH_CALUDE_solve_auction_problem_l1642_164232

def auction_problem (starting_price harry_initial_increase second_bidder_multiplier third_bidder_addition harry_final_increase : ℕ) : Prop :=
  let harry_first_bid := starting_price + harry_initial_increase
  let second_bid := harry_first_bid * second_bidder_multiplier
  let third_bid := second_bid + (harry_first_bid * third_bidder_addition)
  let harry_final_bid := third_bid + harry_final_increase
  harry_final_bid = 4000

theorem solve_auction_problem :
  auction_problem 300 200 2 3 1500 := by sorry

end NUMINAMATH_CALUDE_solve_auction_problem_l1642_164232


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l1642_164234

theorem chord_length_concentric_circles (a b : ℝ) (h1 : a > b) (h2 : a^2 - b^2 = 20) :
  ∃ c : ℝ, c = 4 * Real.sqrt 5 ∧ c^2 / 4 + b^2 = a^2 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l1642_164234


namespace NUMINAMATH_CALUDE_division_of_powers_l1642_164205

theorem division_of_powers (x : ℝ) : 2 * x^5 / ((-x)^3) = -2 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_powers_l1642_164205


namespace NUMINAMATH_CALUDE_zoo_animal_types_l1642_164209

theorem zoo_animal_types :
  let viewing_time : ℕ → ℕ := λ n => 6 * n
  let initial_types : ℕ := 5
  let added_types : ℕ := 4
  let total_time : ℕ := 54
  viewing_time (initial_types + added_types) = total_time :=
by
  sorry

#check zoo_animal_types

end NUMINAMATH_CALUDE_zoo_animal_types_l1642_164209


namespace NUMINAMATH_CALUDE_locus_of_T_and_min_distance_l1642_164270

-- Define the circle A
def circle_A (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 12

-- Define point B
def point_B : ℝ × ℝ := (1, 0)

-- Define the locus Γ
def Γ (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem locus_of_T_and_min_distance :
  ∃ (P : ℝ × ℝ) (T : ℝ × ℝ),
    (circle_A P.1 P.2) ∧
    (∃ (M N : ℝ × ℝ) (H : ℝ × ℝ),
      (Γ M.1 M.2) ∧ (Γ N.1 N.2) ∧
      (H = ((M.1 + N.1) / 2, (M.2 + N.2) / 2)) ∧
      (unit_circle H.1 H.2)) →
    ((∀ (x y : ℝ), (Γ x y) ↔ (x^2 / 3 + y^2 / 2 = 1)) ∧
     (∃ (d : ℝ),
       d = 2 * Real.sqrt 6 / 5 ∧
       ∀ (M N : ℝ × ℝ),
         (Γ M.1 M.2) → (Γ N.1 N.2) →
         (unit_circle ((M.1 + N.1) / 2) ((M.2 + N.2) / 2)) →
         d ≤ Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2))) :=
by sorry


end NUMINAMATH_CALUDE_locus_of_T_and_min_distance_l1642_164270


namespace NUMINAMATH_CALUDE_comic_reconstruction_l1642_164200

theorem comic_reconstruction (pages_per_comic : ℕ) (torn_pages : ℕ) (untorn_comics : ℕ) : 
  pages_per_comic = 45 →
  torn_pages = 2700 →
  untorn_comics = 15 →
  (torn_pages / pages_per_comic + untorn_comics : ℕ) = 75 :=
by sorry

end NUMINAMATH_CALUDE_comic_reconstruction_l1642_164200


namespace NUMINAMATH_CALUDE_solve_laptop_battery_problem_l1642_164241

def laptop_battery_problem (standby_capacity : ℝ) (gaming_capacity : ℝ) 
  (standby_used : ℝ) (gaming_used : ℝ) : Prop :=
  standby_capacity = 10 ∧ 
  gaming_capacity = 2 ∧ 
  standby_used = 4 ∧ 
  gaming_used = 1 ∧ 
  (1 - (standby_used / standby_capacity + gaming_used / gaming_capacity)) * standby_capacity = 1

theorem solve_laptop_battery_problem :
  ∀ standby_capacity gaming_capacity standby_used gaming_used,
  laptop_battery_problem standby_capacity gaming_capacity standby_used gaming_used := by
  sorry

end NUMINAMATH_CALUDE_solve_laptop_battery_problem_l1642_164241


namespace NUMINAMATH_CALUDE_toll_booth_traffic_l1642_164226

theorem toll_booth_traffic (total : ℕ) (mon : ℕ) (tues : ℕ) (wed : ℕ) (thur : ℕ) :
  total = 450 →
  mon = 50 →
  tues = mon →
  wed = 2 * mon →
  thur = wed →
  ∃ (remaining : ℕ), 
    remaining * 3 = total - (mon + tues + wed + thur) ∧
    remaining = 50 :=
by sorry

end NUMINAMATH_CALUDE_toll_booth_traffic_l1642_164226


namespace NUMINAMATH_CALUDE_rational_roots_condition_l1642_164265

theorem rational_roots_condition (p : ℤ) : 
  (∃ x : ℚ, 4 * x^4 + 4 * p * x^3 = (p - 4) * x^2 - 4 * p * x + p) ↔ (p = 0 ∨ p = -1) :=
by sorry

end NUMINAMATH_CALUDE_rational_roots_condition_l1642_164265


namespace NUMINAMATH_CALUDE_peter_erasers_l1642_164283

theorem peter_erasers (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 35 → received = 17 → total = initial + received → total = 52 := by
  sorry

end NUMINAMATH_CALUDE_peter_erasers_l1642_164283


namespace NUMINAMATH_CALUDE_special_function_is_one_l1642_164235

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a > 0 ∧ b > 0 →
    (f (a^2 + b^2) = f a * f b) ∧
    (f (a^2) = (f a)^2)

/-- The main theorem stating that any function satisfying the conditions is constant 1 -/
theorem special_function_is_one (f : ℕ → ℕ) (h : SpecialFunction f) :
  ∀ n : ℕ, n > 0 → f n = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_function_is_one_l1642_164235


namespace NUMINAMATH_CALUDE_honeydews_left_l1642_164282

/-- Represents the problem of Darryl's melon sales --/
structure MelonSales where
  cantaloupe_price : ℕ
  honeydew_price : ℕ
  initial_cantaloupes : ℕ
  initial_honeydews : ℕ
  dropped_cantaloupes : ℕ
  rotten_honeydews : ℕ
  final_cantaloupes : ℕ
  total_revenue : ℕ

/-- Theorem stating the number of honeydews left at the end of the day --/
theorem honeydews_left (sale : MelonSales)
  (h1 : sale.cantaloupe_price = 2)
  (h2 : sale.honeydew_price = 3)
  (h3 : sale.initial_cantaloupes = 30)
  (h4 : sale.initial_honeydews = 27)
  (h5 : sale.dropped_cantaloupes = 2)
  (h6 : sale.rotten_honeydews = 3)
  (h7 : sale.final_cantaloupes = 8)
  (h8 : sale.total_revenue = 85) :
  sale.initial_honeydews - sale.rotten_honeydews -
  ((sale.total_revenue - (sale.initial_cantaloupes - sale.dropped_cantaloupes - sale.final_cantaloupes) * sale.cantaloupe_price) / sale.honeydew_price) = 9 :=
sorry

end NUMINAMATH_CALUDE_honeydews_left_l1642_164282


namespace NUMINAMATH_CALUDE_calculate_expression_l1642_164297

theorem calculate_expression : |-3| - 2 * Real.tan (π / 4) + (-1) ^ 2023 - (Real.sqrt 3 - Real.pi) ^ 0 = -1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1642_164297


namespace NUMINAMATH_CALUDE_solve_equation_l1642_164274

-- Define the * operation
def star (a b : ℝ) : ℝ := 3 * a - b

-- State the theorem
theorem solve_equation : ∃ x : ℝ, star 2 (star 5 x) = 1 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1642_164274


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_shorter_base_l1642_164216

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- Length of the shorter base -/
  shorterBase : ℝ
  /-- Length of the line joining the midpoints of the diagonals -/
  midpointLine : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The midpoint line length is half the difference of the bases -/
  midpointProperty : midpointLine = (longerBase - shorterBase) / 2

/-- Theorem: In an isosceles trapezoid where the line joining the midpoints of the diagonals
    has length 4 and the longer base is 100, the shorter base has length 92 -/
theorem isosceles_trapezoid_shorter_base
  (t : IsoscelesTrapezoid)
  (h1 : t.longerBase = 100)
  (h2 : t.midpointLine = 4) :
  t.shorterBase = 92 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_shorter_base_l1642_164216


namespace NUMINAMATH_CALUDE_test_results_l1642_164222

theorem test_results (total_students : ℕ) (correct_q1 : ℕ) (correct_q2 : ℕ) (not_taken : ℕ)
  (h1 : total_students = 40)
  (h2 : correct_q1 = 30)
  (h3 : correct_q2 = 29)
  (h4 : not_taken = 10)
  : (total_students - not_taken) = correct_q1 ∧ correct_q1 - 1 = correct_q2 := by
  sorry

end NUMINAMATH_CALUDE_test_results_l1642_164222
