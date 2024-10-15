import Mathlib

namespace NUMINAMATH_CALUDE_lcm_gcd_210_396_l1858_185810

theorem lcm_gcd_210_396 :
  (Nat.lcm 210 396 = 4620) ∧ (Nat.gcd 210 396 = 6) := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_210_396_l1858_185810


namespace NUMINAMATH_CALUDE_quadratic_integer_criterion_l1858_185844

/-- A quadratic trinomial ax^2 + bx + c where a, b, and c are real numbers -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic trinomial at a given x -/
def QuadraticTrinomial.eval (q : QuadraticTrinomial) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- Proposition: 2a, a+b, and c are all integers if and only if 
    ax^2 + bx + c takes integer values for all integer x -/
theorem quadratic_integer_criterion (q : QuadraticTrinomial) :
  (∀ x : ℤ, ∃ n : ℤ, q.eval x = n) ↔ 
  (∃ m n p : ℤ, 2 * q.a = m ∧ q.a + q.b = n ∧ q.c = p) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_criterion_l1858_185844


namespace NUMINAMATH_CALUDE_max_square_side_is_40_l1858_185890

def distances_L : List ℕ := [2, 4, 6, 2, 4, 6, 2, 4, 6, 2, 4, 6, 2]
def distances_P : List ℕ := [3, 1, 2, 6, 3, 1, 2, 6, 3, 1, 2, 6, 3, 1]

def max_square_side_length (L : List ℕ) (P : List ℕ) : ℕ := sorry

theorem max_square_side_is_40 :
  max_square_side_length distances_L distances_P = 40 := by sorry

end NUMINAMATH_CALUDE_max_square_side_is_40_l1858_185890


namespace NUMINAMATH_CALUDE_polygon_with_540_degree_sum_is_pentagon_l1858_185839

/-- A polygon with interior angles summing to 540° has 5 sides -/
theorem polygon_with_540_degree_sum_is_pentagon (n : ℕ) : 
  (n - 2) * 180 = 540 → n = 5 := by sorry

end NUMINAMATH_CALUDE_polygon_with_540_degree_sum_is_pentagon_l1858_185839


namespace NUMINAMATH_CALUDE_stationery_cost_l1858_185846

theorem stationery_cost (x y : ℚ) 
  (h1 : 2 * x + 3 * y = 18) 
  (h2 : 3 * x + 2 * y = 22) : 
  x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_l1858_185846


namespace NUMINAMATH_CALUDE_fourth_term_is_plus_minus_three_l1858_185851

/-- A geometric sequence with a_3 = 9 and a_5 = 1 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ 
  a 3 = 9 ∧ 
  a 5 = 1

/-- The fourth term of the geometric sequence is ±3 -/
theorem fourth_term_is_plus_minus_three 
  (a : ℕ → ℝ) 
  (h : geometric_sequence a) : 
  a 4 = 3 ∨ a 4 = -3 :=
sorry

end NUMINAMATH_CALUDE_fourth_term_is_plus_minus_three_l1858_185851


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1858_185801

theorem polynomial_simplification (x : ℝ) :
  (2 * x^10 + 8 * x^9 + 3 * x^8) + (5 * x^12 - x^10 + 2 * x^9 - 5 * x^8 + 4 * x^5 + 6) =
  5 * x^12 + x^10 + 10 * x^9 - 2 * x^8 + 4 * x^5 + 6 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1858_185801


namespace NUMINAMATH_CALUDE_orange_balloons_count_l1858_185854

/-- Given the initial number of orange balloons and the number of additional orange balloons found,
    prove that the total number of orange balloons is equal to their sum. -/
theorem orange_balloons_count 
  (initial_orange : ℝ) 
  (found_orange : ℝ) : 
  initial_orange + found_orange = 11 :=
by
  sorry

#check orange_balloons_count 9 2

end NUMINAMATH_CALUDE_orange_balloons_count_l1858_185854


namespace NUMINAMATH_CALUDE_initial_salt_concentration_l1858_185860

/-- The initial volume of saltwater solution in gallons -/
def x : ℝ := 120

/-- The initial salt concentration as a percentage -/
def C : ℝ := 18.33333333333333

theorem initial_salt_concentration (C : ℝ) :
  (C / 100 * x + 16) / (3 / 4 * x + 8 + 16) = 1 / 3 → C = 18.33333333333333 :=
by sorry

end NUMINAMATH_CALUDE_initial_salt_concentration_l1858_185860


namespace NUMINAMATH_CALUDE_only_piston_and_bottles_are_translations_l1858_185827

/-- Represents a type of motion --/
inductive Motion
| Translation
| Rotation
| Other

/-- Represents the different phenomena described in the problem --/
inductive Phenomenon
| ChildSwinging
| PistonMovement
| PendulumSwinging
| BottlesOnConveyorBelt

/-- Determines the type of motion for a given phenomenon --/
def motionType (p : Phenomenon) : Motion :=
  match p with
  | Phenomenon.ChildSwinging => Motion.Rotation
  | Phenomenon.PistonMovement => Motion.Translation
  | Phenomenon.PendulumSwinging => Motion.Rotation
  | Phenomenon.BottlesOnConveyorBelt => Motion.Translation

/-- Theorem stating that only the piston movement and bottles on conveyor belt are translations --/
theorem only_piston_and_bottles_are_translations :
  (∀ p : Phenomenon, motionType p = Motion.Translation ↔ 
    (p = Phenomenon.PistonMovement ∨ p = Phenomenon.BottlesOnConveyorBelt)) :=
by sorry

end NUMINAMATH_CALUDE_only_piston_and_bottles_are_translations_l1858_185827


namespace NUMINAMATH_CALUDE_pages_per_booklet_l1858_185879

theorem pages_per_booklet (total_booklets : ℕ) (total_pages : ℕ) 
  (h1 : total_booklets = 49) 
  (h2 : total_pages = 441) : 
  total_pages / total_booklets = 9 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_booklet_l1858_185879


namespace NUMINAMATH_CALUDE_two_slices_per_pizza_l1858_185864

/-- Given a total number of pizza slices and a number of pizzas,
    calculate the number of slices per pizza. -/
def slices_per_pizza (total_slices : ℕ) (num_pizzas : ℕ) : ℕ :=
  total_slices / num_pizzas

/-- Prove that given 28 total slices and 14 pizzas, each pizza has 2 slices. -/
theorem two_slices_per_pizza :
  slices_per_pizza 28 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_slices_per_pizza_l1858_185864


namespace NUMINAMATH_CALUDE_tangent_slope_three_points_l1858_185870

theorem tangent_slope_three_points (x y : ℝ) : 
  y = x^3 ∧ (3 * x^2 = 3) → (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_three_points_l1858_185870


namespace NUMINAMATH_CALUDE_fraction_relation_l1858_185872

theorem fraction_relation (q r s t : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 1 / 4) :
  t / q = 8 / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_relation_l1858_185872


namespace NUMINAMATH_CALUDE_initial_state_is_winning_starting_player_wins_starting_player_always_wins_l1858_185895

/-- Represents a pile of matches -/
structure Pile :=
  (count : Nat)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)

/-- Checks if a game state is a winning position for the current player -/
def isWinningPosition (state : GameState) : Prop :=
  ∃ (n m : Nat), n < m ∧
    ∃ (a b c : Nat), 
      state.piles = [Pile.mk (2^n * a), Pile.mk (2^n * b), Pile.mk (2^m * c)] ∧
      Odd a ∧ Odd b ∧ Odd c

/-- The initial game state -/
def initialState : GameState :=
  { piles := [Pile.mk 100, Pile.mk 200, Pile.mk 300] }

/-- Theorem stating that the initial state is a winning position -/
theorem initial_state_is_winning : isWinningPosition initialState := by
  sorry

/-- Theorem stating that the starting player has a winning strategy -/
theorem starting_player_wins (state : GameState) :
  isWinningPosition state → ∃ (nextState : GameState), 
    (∃ (move : GameState → GameState), nextState = move state) ∧
    ¬isWinningPosition nextState := by
  sorry

/-- Main theorem: The starting player wins with correct play -/
theorem starting_player_always_wins : 
  ∃ (strategy : GameState → GameState), 
    ∀ (state : GameState), 
      isWinningPosition state → 
      ¬isWinningPosition (strategy state) := by
  sorry

end NUMINAMATH_CALUDE_initial_state_is_winning_starting_player_wins_starting_player_always_wins_l1858_185895


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1858_185842

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 2) = (a^2 - 3*a + 2) + Complex.I * (a - 2)) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1858_185842


namespace NUMINAMATH_CALUDE_max_alternations_theorem_l1858_185861

/-- Represents a painter's strategy for painting fence sections -/
def PainterStrategy := ℕ → Bool

/-- Represents the state of the fence after painting -/
def FenceState := List Bool

/-- Counts the number of color alternations in a fence state -/
def countAlternations (fence : FenceState) : ℕ := sorry

/-- Simulates the painting process and returns the final fence state -/
def paintFence (strategy1 strategy2 : PainterStrategy) : FenceState := sorry

/-- The maximum number of alternations the first painter can guarantee -/
def maxGuaranteedAlternations : ℕ := sorry

/-- Theorem stating the maximum number of alternations the first painter can guarantee -/
theorem max_alternations_theorem :
  ∀ (strategy2 : PainterStrategy),
  ∃ (strategy1 : PainterStrategy),
  countAlternations (paintFence strategy1 strategy2) ≥ 49 ∧
  maxGuaranteedAlternations = 49 := by sorry

end NUMINAMATH_CALUDE_max_alternations_theorem_l1858_185861


namespace NUMINAMATH_CALUDE_pythago_competition_l1858_185840

theorem pythago_competition (n : ℕ) : 
  (∀ s : ℕ, s ≤ n → ∃! (team : Fin 4 → ℕ), ∀ i j : Fin 4, i ≠ j → team i ≠ team j) →
  (∃ daniel : ℕ, daniel < 50 ∧ 
    (∃ eliza fiona greg : ℕ, 
      eliza = 50 ∧ fiona = 81 ∧ greg = 97 ∧
      daniel < eliza ∧ daniel < fiona ∧ daniel < greg ∧
      (∀ x : ℕ, x ≤ 4*n → (x ≤ daniel ↔ 2*x ≤ 4*n + 1)))) →
  n = 25 := by sorry

end NUMINAMATH_CALUDE_pythago_competition_l1858_185840


namespace NUMINAMATH_CALUDE_divisor_inequality_l1858_185894

theorem divisor_inequality (n : ℕ) (a b c d : ℕ) : 
  (1 < a) → (a < b) → (b < c) → (c < d) → (d < n) →
  (∀ k : ℕ, k ∣ n → (k = 1 ∨ k = a ∨ k = b ∨ k = c ∨ k = d ∨ k = n)) →
  (a ∣ n) → (b ∣ n) → (c ∣ n) → (d ∣ n) →
  b - a ≤ d - c := by
  sorry

#check divisor_inequality

end NUMINAMATH_CALUDE_divisor_inequality_l1858_185894


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l1858_185834

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis :
  let A : Point := { x := 2, y := 3 }
  reflectAcrossXAxis A = { x := 2, y := -3 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l1858_185834


namespace NUMINAMATH_CALUDE_fraction_equality_implies_c_geq_one_l1858_185833

theorem fraction_equality_implies_c_geq_one
  (a b : ℕ+) (c : ℝ)
  (h_c_pos : c > 0)
  (h_eq : (a + 1 : ℝ) / (b + c) = (b : ℝ) / a) :
  c ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_c_geq_one_l1858_185833


namespace NUMINAMATH_CALUDE_inverse_composition_result_l1858_185804

-- Define the functions f and h
variable (f h : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv h_inv : ℝ → ℝ)

-- State the given condition
axiom condition : ∀ x, f_inv (h x) = 6 * x - 4

-- State the theorem to be proved
theorem inverse_composition_result : h_inv (f 3) = 7/6 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_result_l1858_185804


namespace NUMINAMATH_CALUDE_divisibility_by_24_l1858_185847

theorem divisibility_by_24 (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  24 ∣ p^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_24_l1858_185847


namespace NUMINAMATH_CALUDE_equilateral_ABC_l1858_185838

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The incenter of a triangle -/
def incenter (t : Triangle) : Point :=
  sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  sorry

/-- Main theorem -/
theorem equilateral_ABC (A B C I X Y Z : Point) :
  let ABC := Triangle.mk A B C
  let BIC := Triangle.mk B I C
  let CIA := Triangle.mk C I A
  let AIB := Triangle.mk A I B
  let XYZ := Triangle.mk X Y Z
  (I = incenter ABC) →
  (X = incenter BIC) →
  (Y = incenter CIA) →
  (Z = incenter AIB) →
  isEquilateral XYZ →
  isEquilateral ABC :=
by
  sorry

end NUMINAMATH_CALUDE_equilateral_ABC_l1858_185838


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l1858_185822

theorem max_sum_given_constraints (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) :
  x + y ≤ 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l1858_185822


namespace NUMINAMATH_CALUDE_power_inequality_condition_l1858_185809

theorem power_inequality_condition (n : ℤ) : n ∈ ({-2, -1, 0, 1, 2, 3} : Set ℤ) →
  ((-1/2 : ℚ)^n > (-1/5 : ℚ)^n ↔ n = -1 ∨ n = 2) := by sorry

end NUMINAMATH_CALUDE_power_inequality_condition_l1858_185809


namespace NUMINAMATH_CALUDE_points_three_units_from_negative_two_l1858_185845

theorem points_three_units_from_negative_two :
  ∃! (S : Set ℝ), (∀ x ∈ S, |x - (-2)| = 3) ∧ S = {-5, 1} := by
  sorry

end NUMINAMATH_CALUDE_points_three_units_from_negative_two_l1858_185845


namespace NUMINAMATH_CALUDE_existence_of_three_similar_numbers_l1858_185897

def is_1995_digit (n : ℕ) : Prop := n ≥ 10^1994 ∧ n < 10^1995

def composed_of_4_5_9 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 4 ∨ d = 5 ∨ d = 9

def similar (a b : ℕ) : Prop :=
  ∀ d : ℕ, (d ∈ a.digits 10 ↔ d ∈ b.digits 10)

theorem existence_of_three_similar_numbers :
  ∃ (A B C : ℕ),
    is_1995_digit A ∧
    is_1995_digit B ∧
    is_1995_digit C ∧
    composed_of_4_5_9 A ∧
    composed_of_4_5_9 B ∧
    composed_of_4_5_9 C ∧
    similar A B ∧
    similar B C ∧
    similar A C ∧
    A + B = C :=
  sorry

end NUMINAMATH_CALUDE_existence_of_three_similar_numbers_l1858_185897


namespace NUMINAMATH_CALUDE_train_interval_l1858_185828

/-- Represents a metro station -/
inductive Station : Type
| Taganskaya : Station
| Kievskaya : Station

/-- Represents a direction of travel -/
inductive Direction : Type
| Clockwise : Direction
| Counterclockwise : Direction

/-- Represents the metro system -/
structure MetroSystem where
  northern_route_time : ℝ
  southern_route_time : ℝ
  train_delay : ℝ
  trip_time_difference : ℝ

/-- Calculate the expected travel time between stations -/
def expected_travel_time (m : MetroSystem) (p : ℝ) : ℝ :=
  m.southern_route_time * p + m.northern_route_time * (1 - p)

/-- Theorem: The interval between trains in one direction is 3 minutes -/
theorem train_interval (m : MetroSystem) 
  (h1 : m.northern_route_time = 17)
  (h2 : m.southern_route_time = 11)
  (h3 : m.train_delay = 5/4)
  (h4 : m.trip_time_difference = 1)
  : ∃ (T : ℝ), T = 3 ∧ 
    ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧
    expected_travel_time m p = expected_travel_time m (1-p) - m.trip_time_difference ∧
    T * (1 - p) = m.train_delay := by
  sorry

end NUMINAMATH_CALUDE_train_interval_l1858_185828


namespace NUMINAMATH_CALUDE_hyperbola_max_ratio_hyperbola_max_ratio_achievable_l1858_185829

theorem hyperbola_max_ratio (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_focal : c^2 = a^2 + b^2) : 
  (a + b) / c ≤ Real.sqrt 2 :=
sorry

theorem hyperbola_max_ratio_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ c^2 = a^2 + b^2 ∧ (a + b) / c = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_max_ratio_hyperbola_max_ratio_achievable_l1858_185829


namespace NUMINAMATH_CALUDE_z_equals_negative_four_l1858_185874

theorem z_equals_negative_four (x y z : ℤ) : x = 2 → y = x^2 - 5 → z = y^2 - 5 → z = -4 := by
  sorry

end NUMINAMATH_CALUDE_z_equals_negative_four_l1858_185874


namespace NUMINAMATH_CALUDE_triangle_is_acute_l1858_185896

theorem triangle_is_acute (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C)
  (h_ratio : (Real.sin A + Real.sin B) / (Real.sin B + Real.sin C) = 9 / 11 ∧
             (Real.sin B + Real.sin C) / (Real.sin C + Real.sin A) = 11 / 10) :
  A < π/2 ∧ B < π/2 ∧ C < π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_acute_l1858_185896


namespace NUMINAMATH_CALUDE_min_square_side_for_9x21_l1858_185875

/-- The minimum side length of a square that can contain 9x21 rectangles without rotation and overlap -/
def min_square_side (width : ℕ) (length : ℕ) : ℕ :=
  Nat.lcm width length

/-- Theorem stating that the minimum side length for 9x21 rectangles is 63 -/
theorem min_square_side_for_9x21 :
  min_square_side 9 21 = 63 := by sorry

end NUMINAMATH_CALUDE_min_square_side_for_9x21_l1858_185875


namespace NUMINAMATH_CALUDE_nine_numbers_system_solution_l1858_185835

theorem nine_numbers_system_solution (n : ℕ) (S : Finset ℕ) 
  (h₁ : n ≥ 3)
  (h₂ : S ⊆ Finset.range (n^3 + 1))
  (h₃ : S.card = 3 * n^2) :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ) (x y z : ℤ),
    a₁ ∈ S ∧ a₂ ∈ S ∧ a₃ ∈ S ∧ a₄ ∈ S ∧ a₅ ∈ S ∧ a₆ ∈ S ∧ a₇ ∈ S ∧ a₈ ∈ S ∧ a₉ ∈ S ∧
    a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧ a₁ ≠ a₈ ∧ a₁ ≠ a₉ ∧
    a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧ a₂ ≠ a₈ ∧ a₂ ≠ a₉ ∧
    a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧ a₃ ≠ a₈ ∧ a₃ ≠ a₉ ∧
    a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧ a₄ ≠ a₈ ∧ a₄ ≠ a₉ ∧
    a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧ a₅ ≠ a₈ ∧ a₅ ≠ a₉ ∧
    a₆ ≠ a₇ ∧ a₆ ≠ a₈ ∧ a₆ ≠ a₉ ∧
    a₇ ≠ a₈ ∧ a₇ ≠ a₉ ∧
    a₈ ≠ a₉ ∧
    x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    (a₁ : ℤ) * x + (a₂ : ℤ) * y + (a₃ : ℤ) * z = 0 ∧
    (a₄ : ℤ) * x + (a₅ : ℤ) * y + (a₆ : ℤ) * z = 0 ∧
    (a₇ : ℤ) * x + (a₈ : ℤ) * y + (a₉ : ℤ) * z = 0 := by
  sorry

end NUMINAMATH_CALUDE_nine_numbers_system_solution_l1858_185835


namespace NUMINAMATH_CALUDE_smallest_group_size_fifty_nine_satisfies_conditions_fewest_students_l1858_185878

theorem smallest_group_size (N : ℕ) : 
  (N % 5 = 2) ∧ (N % 6 = 3) ∧ (N % 8 = 4) → N ≥ 59 :=
by sorry

theorem fifty_nine_satisfies_conditions : 
  (59 % 5 = 2) ∧ (59 % 6 = 3) ∧ (59 % 8 = 4) :=
by sorry

theorem fewest_students : 
  ∃ (N : ℕ), (N % 5 = 2) ∧ (N % 6 = 3) ∧ (N % 8 = 4) ∧ 
  (∀ (M : ℕ), (M % 5 = 2) ∧ (M % 6 = 3) ∧ (M % 8 = 4) → M ≥ N) ∧
  N = 59 :=
by sorry

end NUMINAMATH_CALUDE_smallest_group_size_fifty_nine_satisfies_conditions_fewest_students_l1858_185878


namespace NUMINAMATH_CALUDE_triangle_count_l1858_185898

/-- The number of points on the circumference of the circle -/
def n : ℕ := 7

/-- The number of points needed to form a triangle -/
def k : ℕ := 3

/-- The number of different triangles that can be formed -/
def num_triangles : ℕ := Nat.choose n k

theorem triangle_count : num_triangles = 35 := by sorry

end NUMINAMATH_CALUDE_triangle_count_l1858_185898


namespace NUMINAMATH_CALUDE_primary_school_ages_l1858_185866

theorem primary_school_ages (x y : ℕ) : 
  7 ≤ x ∧ x ≤ 13 ∧ 7 ≤ y ∧ y ≤ 13 →
  (x + y) * (x - y) = 63 →
  x = 12 ∧ y = 9 := by
sorry

end NUMINAMATH_CALUDE_primary_school_ages_l1858_185866


namespace NUMINAMATH_CALUDE_square_area_perimeter_ratio_l1858_185821

theorem square_area_perimeter_ratio : 
  ∀ (s1 s2 : ℝ), s1 > 0 ∧ s2 > 0 →
  (s1^2 : ℝ) / (s2^2 : ℝ) = 49 / 64 →
  (4 * s1) / (4 * s2) = 7 / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_square_area_perimeter_ratio_l1858_185821


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1858_185802

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.sqrt (18 * x) * Real.sqrt (2 * x) * Real.sqrt (25 * x) * Real.sqrt (5 * x) = 50 ∧ 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ |x - 0.8632| < ε := by
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1858_185802


namespace NUMINAMATH_CALUDE_cat_cafe_ratio_l1858_185819

/-- The number of cats in Cat Cafe Cool -/
def cool_cats : ℕ := 5

/-- The number of cats in Cat Cafe Paw -/
def paw_cats : ℕ := 10

/-- The number of cats in Cat Cafe Meow -/
def meow_cats : ℕ := 3 * paw_cats

/-- The total number of cats in Cat Cafe Meow and Cat Cafe Paw -/
def total_cats : ℕ := 40

/-- The theorem stating the ratio of cats between Cat Cafe Paw and Cat Cafe Cool -/
theorem cat_cafe_ratio : paw_cats / cool_cats = 2 := by
  sorry

end NUMINAMATH_CALUDE_cat_cafe_ratio_l1858_185819


namespace NUMINAMATH_CALUDE_alex_final_silver_tokens_l1858_185880

/-- Represents the number of tokens Alex has -/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents a token exchange booth -/
structure Booth where
  red_in : ℕ
  blue_in : ℕ
  red_out : ℕ
  blue_out : ℕ
  silver_out : ℕ

/-- Applies a single exchange at a booth -/
def apply_exchange (tokens : TokenCount) (booth : Booth) : TokenCount :=
  { red := tokens.red - booth.red_in + booth.red_out,
    blue := tokens.blue - booth.blue_in + booth.blue_out,
    silver := tokens.silver + booth.silver_out }

/-- Checks if an exchange is possible -/
def can_exchange (tokens : TokenCount) (booth : Booth) : Prop :=
  tokens.red ≥ booth.red_in ∧ tokens.blue ≥ booth.blue_in

/-- The final state after all possible exchanges -/
def final_state (initial : TokenCount) (booth1 booth2 : Booth) : TokenCount :=
  sorry  -- The implementation would go here

/-- Theorem stating that Alex will end up with 58 silver tokens -/
theorem alex_final_silver_tokens :
  let initial := TokenCount.mk 100 50 0
  let booth1 := Booth.mk 3 0 0 1 2
  let booth2 := Booth.mk 0 4 1 0 1
  (final_state initial booth1 booth2).silver = 58 := by
  sorry


end NUMINAMATH_CALUDE_alex_final_silver_tokens_l1858_185880


namespace NUMINAMATH_CALUDE_passengers_at_terminal_l1858_185884

/-- Represents the number of stations on the bus route. -/
def num_stations : ℕ := 8

/-- Represents the number of people who boarded the bus at the first 6 stations. -/
def passengers_boarded : ℕ := 100

/-- Represents the number of people who got off at all stations except the terminal station. -/
def passengers_got_off : ℕ := 80

/-- Theorem stating that the number of passengers who boarded at the first 6 stations
    and got off at the terminal station is 20. -/
theorem passengers_at_terminal : ℕ := by
  sorry

#check passengers_at_terminal

end NUMINAMATH_CALUDE_passengers_at_terminal_l1858_185884


namespace NUMINAMATH_CALUDE_units_digit_fourth_power_not_seven_l1858_185812

theorem units_digit_fourth_power_not_seven :
  ∀ n : ℕ, (n^4 % 10) ≠ 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_fourth_power_not_seven_l1858_185812


namespace NUMINAMATH_CALUDE_magnitude_of_AD_is_two_l1858_185803

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

end NUMINAMATH_CALUDE_magnitude_of_AD_is_two_l1858_185803


namespace NUMINAMATH_CALUDE_exterior_angle_measure_l1858_185814

theorem exterior_angle_measure (n : ℕ) (h : n > 2) :
  (n - 2) * 180 = 1260 →
  360 / n = 40 := by
sorry

end NUMINAMATH_CALUDE_exterior_angle_measure_l1858_185814


namespace NUMINAMATH_CALUDE_amethyst_bead_count_l1858_185823

/-- Proves the number of amethyst beads in a specific necklace configuration -/
theorem amethyst_bead_count (total : ℕ) (turquoise : ℕ) (amethyst : ℕ) : 
  total = 40 → 
  turquoise = 19 → 
  total = amethyst + 2 * amethyst + turquoise → 
  amethyst = 7 := by
  sorry

end NUMINAMATH_CALUDE_amethyst_bead_count_l1858_185823


namespace NUMINAMATH_CALUDE_find_a_l1858_185899

theorem find_a : ∃ a : ℚ, (a + 3) / 4 = (2 * a - 3) / 7 + 1 → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l1858_185899


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l1858_185855

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y, x < y → f x > f y) →  -- f is decreasing on ℝ
  f (3 * a) < f (-2 * a + 10) →
  a > 2 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l1858_185855


namespace NUMINAMATH_CALUDE_sin_225_plus_alpha_l1858_185881

theorem sin_225_plus_alpha (α : ℝ) (h : Real.sin (π/4 + α) = 5/13) :
  Real.sin (5*π/4 + α) = -5/13 := by
  sorry

end NUMINAMATH_CALUDE_sin_225_plus_alpha_l1858_185881


namespace NUMINAMATH_CALUDE_circumradius_arithmetic_angles_max_inradius_arithmetic_sides_max_inradius_achieved_l1858_185889

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The circumradius of a triangle -/
def circumradius (t : Triangle) : ℝ := sorry

/-- The inradius of a triangle -/
def inradius (t : Triangle) : ℝ := sorry

/-- Theorem: Circumradius of triangle with arithmetic sequence angles -/
theorem circumradius_arithmetic_angles (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : ∃ k : ℝ, t.B = t.A + k ∧ t.C = t.B + k) : 
  circumradius t = 2 * Real.sqrt 3 / 3 := by sorry

/-- Theorem: Maximum inradius of triangle with arithmetic sequence sides -/
theorem max_inradius_arithmetic_sides (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : ∃ k : ℝ, t.b = t.a + k ∧ t.c = t.b + k) :
  inradius t ≤ Real.sqrt 3 / 3 := by sorry

/-- Corollary: The maximum inradius is achieved -/
theorem max_inradius_achieved (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : ∃ k : ℝ, t.b = t.a + k ∧ t.c = t.b + k) :
  ∃ t' : Triangle, t'.b = 2 ∧ (∃ k : ℝ, t'.b = t'.a + k ∧ t'.c = t'.b + k) ∧ 
  inradius t' = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_circumradius_arithmetic_angles_max_inradius_arithmetic_sides_max_inradius_achieved_l1858_185889


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1858_185808

/-- The line 4x + 7y + 49 = 0 is tangent to the parabola y^2 = 16x -/
theorem line_tangent_to_parabola :
  ∃! (x y : ℝ), 4 * x + 7 * y + 49 = 0 ∧ y^2 = 16 * x := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1858_185808


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1858_185873

theorem quadratic_no_real_roots (m : ℝ) : 
  (∀ x : ℝ, (m + 2) * x^2 - x + m ≠ 0) ↔ 
  (m < -1 - Real.sqrt 5 / 2 ∨ m > -1 + Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1858_185873


namespace NUMINAMATH_CALUDE_pear_seed_average_l1858_185848

theorem pear_seed_average (total_seeds : ℕ) (apple_seeds : ℕ) (grape_seeds : ℕ)
  (num_apples : ℕ) (num_pears : ℕ) (num_grapes : ℕ) (seeds_needed : ℕ) :
  total_seeds = 60 →
  apple_seeds = 6 →
  grape_seeds = 3 →
  num_apples = 4 →
  num_pears = 3 →
  num_grapes = 9 →
  seeds_needed = 3 →
  ∃ (pear_seeds : ℕ), pear_seeds = 2 ∧
    num_apples * apple_seeds + num_pears * pear_seeds + num_grapes * grape_seeds = total_seeds - seeds_needed :=
by sorry

end NUMINAMATH_CALUDE_pear_seed_average_l1858_185848


namespace NUMINAMATH_CALUDE_min_value_zero_l1858_185869

open Real

/-- The quadratic expression in x and y with parameter c -/
def f (c x y : ℝ) : ℝ :=
  3 * x^2 - 4 * c * x * y + (2 * c^2 + 1) * y^2 - 6 * x - 3 * y + 5

/-- The theorem stating the condition for minimum value of f to be 0 -/
theorem min_value_zero (c : ℝ) :
  (∀ x y : ℝ, f c x y ≥ 0) ∧ (∃ x y : ℝ, f c x y = 0) ↔ c = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_zero_l1858_185869


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l1858_185830

theorem three_digit_number_proof :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n / 100 = 1 ∧
  (n % 100 * 10 + 1) - n = 9 * (10 : ℝ) ∧ n = 121 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l1858_185830


namespace NUMINAMATH_CALUDE_sine_function_properties_l1858_185888

/-- The function f we're analyzing -/
noncomputable def f (x : ℝ) : ℝ := sorry

/-- The angular frequency ω -/
noncomputable def ω : ℝ := sorry

/-- The phase shift φ -/
noncomputable def φ : ℝ := sorry

/-- The constant M -/
noncomputable def M : ℝ := sorry

/-- Theorem stating the properties of f and the conclusion -/
theorem sine_function_properties :
  (∃ A : ℝ, ∀ x : ℝ, f x ≤ f A) ∧  -- A is a highest point
  (ω > 0) ∧
  (0 < φ ∧ φ < 2 * Real.pi) ∧
  (∃ B C : ℝ, B < C ∧  -- B and C are adjacent centers of symmetry
    (∀ x : ℝ, f (B + x) = f (B - x)) ∧
    (∀ x : ℝ, f (C + x) = f (C - x)) ∧
    (C - B = Real.pi / ω)) ∧
  ((C - B) * (f A) / 2 = 1 / 2) ∧  -- Area of triangle ABC is 1/2
  (M > 0 ∧ ∀ x : ℝ, f (x + M) = M * f (-x)) →  -- Functional equation
  (∀ x : ℝ, f x = -Real.sin (Real.pi * x)) := by
sorry

end NUMINAMATH_CALUDE_sine_function_properties_l1858_185888


namespace NUMINAMATH_CALUDE_complex_sum_cube_ratio_l1858_185816

theorem complex_sum_cube_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_sq_diff : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 33 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_cube_ratio_l1858_185816


namespace NUMINAMATH_CALUDE_cash_drawer_value_l1858_185824

/-- Calculates the total value of bills in a cash drawer given the total number of bills,
    the number of 5-dollar bills, and assuming the rest are 20-dollar bills. -/
def total_value (total_bills : ℕ) (five_dollar_bills : ℕ) : ℕ :=
  let twenty_dollar_bills := total_bills - five_dollar_bills
  5 * five_dollar_bills + 20 * twenty_dollar_bills

/-- Theorem stating that given 54 bills in total with 20 5-dollar bills,
    the total value is $780. -/
theorem cash_drawer_value :
  total_value 54 20 = 780 := by
  sorry

#eval total_value 54 20  -- Should output 780

end NUMINAMATH_CALUDE_cash_drawer_value_l1858_185824


namespace NUMINAMATH_CALUDE_balance_scale_l1858_185858

/-- The weight of a green ball in terms of blue balls -/
def green_to_blue : ℚ := 2

/-- The weight of a yellow ball in terms of blue balls -/
def yellow_to_blue : ℚ := 8/3

/-- The weight of a white ball in terms of blue balls -/
def white_to_blue : ℚ := 5/3

/-- The number of green balls on the scale -/
def num_green : ℕ := 3

/-- The number of yellow balls on the scale -/
def num_yellow : ℕ := 3

/-- The number of white balls on the scale -/
def num_white : ℕ := 3

theorem balance_scale : 
  (num_green : ℚ) * green_to_blue + 
  (num_yellow : ℚ) * yellow_to_blue + 
  (num_white : ℚ) * white_to_blue = 19 := by sorry

end NUMINAMATH_CALUDE_balance_scale_l1858_185858


namespace NUMINAMATH_CALUDE_ab_geq_2_sufficient_not_necessary_l1858_185892

theorem ab_geq_2_sufficient_not_necessary :
  (∀ a b : ℝ, a * b ≥ 2 → a^2 + b^2 ≥ 4) ∧
  (∃ a b : ℝ, a^2 + b^2 ≥ 4 ∧ a * b < 2) := by
  sorry

end NUMINAMATH_CALUDE_ab_geq_2_sufficient_not_necessary_l1858_185892


namespace NUMINAMATH_CALUDE_system_solution_l1858_185850

theorem system_solution (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) :
  (∃ x y : ℝ, a₁ * x + b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂ ∧ x = 8 ∧ y = 5) →
  (∃ x y : ℝ, 4 * a₁ * x - 5 * b₁ * y = 3 * c₁ ∧ 4 * a₂ * x - 5 * b₂ * y = 3 * c₂ ∧ x = 6 ∧ y = -3) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1858_185850


namespace NUMINAMATH_CALUDE_function_properties_l1858_185820

/-- Given function f with properties as described -/
def f (x : ℝ) : ℝ := sorry

/-- ω is a positive real number -/
def ω : ℝ := sorry

/-- φ is a real number between 0 and π -/
def φ : ℝ := sorry

theorem function_properties (x α : ℝ) :
  ω > 0 ∧
  0 ≤ φ ∧ φ ≤ π ∧
  (∀ x, f x = Real.sin (ω * x + φ)) ∧
  (∀ x, f x = f (-x)) ∧
  (∃ k : ℤ, ∀ x, f (x + π) = f x) ∧
  Real.sin α + f α = 2/3 →
  (f = Real.cos) ∧
  ((Real.sqrt 2 * Real.sin (2*α - π/4) + 1) / (1 + Real.tan α) = 5/9) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1858_185820


namespace NUMINAMATH_CALUDE_lumberjack_problem_l1858_185882

theorem lumberjack_problem (logs_per_tree : ℕ) (firewood_per_log : ℕ) (total_firewood : ℕ) :
  logs_per_tree = 4 →
  firewood_per_log = 5 →
  total_firewood = 500 →
  (total_firewood / firewood_per_log) / logs_per_tree = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_lumberjack_problem_l1858_185882


namespace NUMINAMATH_CALUDE_x_fifth_minus_five_x_equals_3100_l1858_185800

theorem x_fifth_minus_five_x_equals_3100 (x : ℝ) (h : x = 5) : x^5 - 5*x = 3100 := by
  sorry

end NUMINAMATH_CALUDE_x_fifth_minus_five_x_equals_3100_l1858_185800


namespace NUMINAMATH_CALUDE_power_24_in_terms_of_P_l1858_185856

theorem power_24_in_terms_of_P (a b : ℕ) (P : ℝ) (h_P : P = 2^a) : 24^(a*b) = P^(3*b) * 3^(a*b) := by
  sorry

end NUMINAMATH_CALUDE_power_24_in_terms_of_P_l1858_185856


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_greater_than_one_l1858_185867

theorem quadratic_always_positive_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_greater_than_one_l1858_185867


namespace NUMINAMATH_CALUDE_curve_is_parabola_l1858_185843

-- Define the curve
def curve (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + y^2) = |3*x - 4*y + 2| / 5

-- Define the fixed point F
def F : ℝ × ℝ := (2, 0)

-- Define the line
def line (x y : ℝ) : Prop :=
  3*x - 4*y + 2 = 0

-- Theorem statement
theorem curve_is_parabola :
  ∃ (f : ℝ × ℝ) (l : ℝ → ℝ → Prop),
    (∀ x y, curve x y ↔ 
      (Real.sqrt ((x - f.1)^2 + (y - f.2)^2) = 
       Real.sqrt ((3*x - 4*y + 2)^2) / 5)) ∧
    (f = F) ∧
    (∀ x y, l x y ↔ line x y) ∧
    (¬ l F.1 F.2) :=
  sorry

end NUMINAMATH_CALUDE_curve_is_parabola_l1858_185843


namespace NUMINAMATH_CALUDE_line_through_points_l1858_185841

theorem line_through_points (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ n : ℤ, (b / a : ℝ) = n) (h4 : ∃ m : ℝ, ∀ x y : ℝ, y = k * x + m → (x = a ∧ y = a) ∨ (x = b ∧ y = 8 * b)) :
  k = 9 ∨ k = 15 :=
sorry

end NUMINAMATH_CALUDE_line_through_points_l1858_185841


namespace NUMINAMATH_CALUDE_lcm_12_20_l1858_185805

theorem lcm_12_20 : Nat.lcm 12 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_20_l1858_185805


namespace NUMINAMATH_CALUDE_probability_ratio_equals_ways_ratio_l1858_185806

def number_of_balls : ℕ := 20
def number_of_bins : ℕ := 5

def distribution_p : List ℕ := [3, 6, 4, 4, 4]
def distribution_q : List ℕ := [4, 4, 4, 4, 4]

def ways_to_distribute (dist : List ℕ) : ℕ :=
  sorry

theorem probability_ratio_equals_ways_ratio :
  let p := (ways_to_distribute distribution_p : ℚ) / number_of_balls ^ number_of_bins
  let q := (ways_to_distribute distribution_q : ℚ) / number_of_balls ^ number_of_bins
  p / q = (ways_to_distribute distribution_p : ℚ) / (ways_to_distribute distribution_q) :=
by sorry

end NUMINAMATH_CALUDE_probability_ratio_equals_ways_ratio_l1858_185806


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1858_185863

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ (b : ℝ), (1 - a * Complex.I) / (1 + Complex.I) = b * Complex.I) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1858_185863


namespace NUMINAMATH_CALUDE_infinitely_many_nondivisible_l1858_185893

theorem infinitely_many_nondivisible (a b : ℕ) : 
  Set.Infinite {n : ℕ | ¬(n^b + 1 ∣ a^n + 1)} := by
sorry

end NUMINAMATH_CALUDE_infinitely_many_nondivisible_l1858_185893


namespace NUMINAMATH_CALUDE_f_value_at_3_l1858_185883

theorem f_value_at_3 (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = a * x^2 + b * x + 2) →
  f 1 = 3 →
  f 2 = 12 →
  f 3 = 29 := by
sorry

end NUMINAMATH_CALUDE_f_value_at_3_l1858_185883


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1858_185885

theorem polar_to_rectangular_conversion :
  let r : ℝ := 3 * Real.sqrt 2
  let θ : ℝ := 5 * Real.pi / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = -3 * Real.sqrt 6 / 2) ∧ (y = 3 * Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1858_185885


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l1858_185859

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_t_value :
  ∀ t : ℝ,
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (4, t)
  parallel a b → t = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l1858_185859


namespace NUMINAMATH_CALUDE_zealand_has_one_fifth_l1858_185865

/-- Represents the amount of money each person has -/
structure Money where
  wanda : ℚ
  xander : ℚ
  yusuf : ℚ
  zealand : ℚ

/-- The initial state of money distribution -/
def initial_money : Money :=
  { wanda := 6, xander := 5, yusuf := 4, zealand := 0 }

/-- The state of money after Zealand receives money from others -/
def final_money : Money :=
  { wanda := 5, xander := 4, yusuf := 3, zealand := 3 }

/-- The fraction of money Zealand has at the end -/
def zealand_fraction (m : Money) : ℚ :=
  m.zealand / (m.wanda + m.xander + m.yusuf + m.zealand)

/-- Theorem stating that Zealand ends up with 1/5 of the total money -/
theorem zealand_has_one_fifth :
  zealand_fraction final_money = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_zealand_has_one_fifth_l1858_185865


namespace NUMINAMATH_CALUDE_history_score_calculation_l1858_185832

def geography_score : ℕ := 50
def math_score : ℕ := 70
def english_score : ℕ := 66
def total_score : ℕ := 248

theorem history_score_calculation :
  total_score - (geography_score + math_score + english_score) =
  total_score - geography_score - math_score - english_score :=
by sorry

end NUMINAMATH_CALUDE_history_score_calculation_l1858_185832


namespace NUMINAMATH_CALUDE_storage_unit_paint_area_l1858_185807

/-- Represents a rectangular storage unit with windows --/
structure StorageUnit where
  length : ℝ
  width : ℝ
  height : ℝ
  windowCount : ℕ
  windowLength : ℝ
  windowWidth : ℝ

/-- Calculates the total area to be painted in the storage unit --/
def totalPaintArea (unit : StorageUnit) : ℝ :=
  let wallArea := 2 * (unit.length * unit.height + unit.width * unit.height)
  let ceilingArea := unit.length * unit.width
  let windowArea := unit.windowCount * (unit.windowLength * unit.windowWidth)
  wallArea + ceilingArea - windowArea

/-- Theorem stating that the total paint area for the given storage unit is 1020 square yards --/
theorem storage_unit_paint_area :
  let unit : StorageUnit := {
    length := 15,
    width := 12,
    height := 8,
    windowCount := 2,
    windowLength := 3,
    windowWidth := 4
  }
  totalPaintArea unit = 1020 := by sorry

end NUMINAMATH_CALUDE_storage_unit_paint_area_l1858_185807


namespace NUMINAMATH_CALUDE_cos_beta_value_l1858_185815

theorem cos_beta_value (α β : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.tan α = 2) (h4 : Real.sin (α + β) = Real.sqrt 2 / 2) :
  Real.cos β = Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_beta_value_l1858_185815


namespace NUMINAMATH_CALUDE_lowest_price_type_a_l1858_185826

/-- Calculates the final price of a pet food type given its MSRP, regular discount, additional discount, and sales tax rate -/
def finalPrice (msrp : ℝ) (regularDiscount : ℝ) (additionalDiscount : ℝ) (salesTax : ℝ) : ℝ :=
  let discountedPrice := msrp * (1 - regularDiscount)
  let furtherDiscountedPrice := discountedPrice * (1 - additionalDiscount)
  furtherDiscountedPrice * (1 + salesTax)

theorem lowest_price_type_a (msrp_a msrp_b msrp_c : ℝ) :
  msrp_a = 45 ∧ msrp_b = 55 ∧ msrp_c = 50 →
  finalPrice msrp_a 0.15 0.20 0.07 < finalPrice msrp_b 0.25 0.15 0.07 ∧
  finalPrice msrp_a 0.15 0.20 0.07 < finalPrice msrp_c 0.30 0.10 0.07 :=
by sorry

end NUMINAMATH_CALUDE_lowest_price_type_a_l1858_185826


namespace NUMINAMATH_CALUDE_bernoulli_joint_distribution_theorem_bernoulli_independence_theorem_l1858_185849

/-- Bernoulli random variable -/
structure BernoulliRV where
  p : ℝ
  hp : 0 ≤ p ∧ p ≤ 1

/-- Joint distribution of two Bernoulli random variables -/
structure JointDistribution (X Y : BernoulliRV) where
  pxy : ℝ × ℝ → ℝ
  sum_to_one : (pxy (0, 0)) + (pxy (0, 1)) + (pxy (1, 0)) + (pxy (1, 1)) = 1

/-- Covariance of two Bernoulli random variables -/
def cov (X Y : BernoulliRV) : ℝ := sorry

/-- Main theorem -/
theorem bernoulli_joint_distribution_theorem (X Y : BernoulliRV) :
  ∃! (j : JointDistribution X Y),
    (j.pxy (1, 1) = cov X Y + X.p * Y.p) ∧
    (j.pxy (0, 1) = Y.p - (cov X Y + X.p * Y.p)) ∧
    (j.pxy (1, 0) = X.p - (cov X Y + X.p * Y.p)) ∧
    (j.pxy (0, 0) = 1 - X.p - Y.p + (cov X Y + X.p * Y.p)) :=
  sorry

/-- Independence theorem -/
theorem bernoulli_independence_theorem (X Y : BernoulliRV) (j : JointDistribution X Y) :
  (∀ x y, j.pxy (x, y) = (if x = 1 then X.p else 1 - X.p) * (if y = 1 then Y.p else 1 - Y.p)) ↔
  cov X Y = 0 :=
  sorry

end NUMINAMATH_CALUDE_bernoulli_joint_distribution_theorem_bernoulli_independence_theorem_l1858_185849


namespace NUMINAMATH_CALUDE_cassandra_apple_pie_l1858_185868

/-- The number of apples in each slice of pie -/
def apples_per_slice (total_apples : ℕ) (num_pies : ℕ) (slices_per_pie : ℕ) : ℚ :=
  (total_apples : ℚ) / (num_pies * slices_per_pie)

/-- Cassandra's apple pie problem -/
theorem cassandra_apple_pie :
  let total_apples : ℕ := 4 * 12  -- 4 dozen
  let num_pies : ℕ := 4
  let slices_per_pie : ℕ := 6
  apples_per_slice total_apples num_pies slices_per_pie = 2 := by
sorry

end NUMINAMATH_CALUDE_cassandra_apple_pie_l1858_185868


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1858_185811

/-- Given two perpendicular lines with direction vectors (2, 5) and (b, -3), prove that b = 15/2 -/
theorem perpendicular_lines_b_value (b : ℝ) : 
  let v₁ : Fin 2 → ℝ := ![2, 5]
  let v₂ : Fin 2 → ℝ := ![b, -3]
  (∀ i : Fin 2, v₁ i * v₂ i = 0) → b = 15/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1858_185811


namespace NUMINAMATH_CALUDE_exists_close_vertices_l1858_185817

/-- A regular polygon with 2n+1 sides inscribed in a unit circle -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n+1) → ℝ × ℝ
  is_regular : ∀ i : Fin (2*n+1), norm (vertices i) = 1

/-- A point inside the polygon -/
def InsidePoint (n : ℕ) (poly : RegularPolygon n) := { p : ℝ × ℝ // norm p < 1 }

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := norm (p.1 - q.1, p.2 - q.2)

/-- Statement of the theorem -/
theorem exists_close_vertices (n : ℕ) :
  ∃ α : ℝ, α > 0 ∧
  ∀ (poly : RegularPolygon n) (p : InsidePoint n poly),
  ∃ (i j : Fin (2*n+1)), i ≠ j ∧
  |distance p.val (poly.vertices i) - distance p.val (poly.vertices j)| < 1/n - α/n^3 :=
sorry

end NUMINAMATH_CALUDE_exists_close_vertices_l1858_185817


namespace NUMINAMATH_CALUDE_polynomial_degree_is_12_l1858_185836

/-- The degree of a polynomial (x^5 + ax^8 + bx^2 + c)(y^3 + dy^2 + e)(z + f) -/
def polynomial_degree (a b c d e f : ℝ) : ℕ :=
  let p1 := fun (x : ℝ) => x^5 + a*x^8 + b*x^2 + c
  let p2 := fun (y : ℝ) => y^3 + d*y^2 + e
  let p3 := fun (z : ℝ) => z + f
  let product := fun (x y z : ℝ) => p1 x * p2 y * p3 z
  12

theorem polynomial_degree_is_12 (a b c d e f : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
    (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0) :
    polynomial_degree a b c d e f = 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_is_12_l1858_185836


namespace NUMINAMATH_CALUDE_center_is_five_l1858_185891

/-- Represents a 3x3 grid with numbers from 1 to 9 --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if two positions in the grid are adjacent --/
def adjacent (p q : Fin 3 × Fin 3) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ q.2.val + 1 = p.2.val)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ q.1.val + 1 = p.1.val))

/-- Checks if two numbers are consecutive --/
def consecutive (m n : Fin 9) : Prop :=
  m.val + 1 = n.val ∨ n.val + 1 = m.val

/-- Main theorem --/
theorem center_is_five (g : Grid) : 
  (∀ i j, g i j ≠ g i j → False) → -- Each number is used once
  (∀ i j k l, consecutive (g i j) (g k l) → adjacent (i, j) (k, l)) → -- Consecutive numbers are adjacent
  g 0 0 = 1 → g 0 2 = 3 → g 2 0 = 5 → g 2 2 = 7 → -- Corner numbers are 2, 4, 6, 8
  g 1 1 = 4 -- Center is 5
  := by sorry

end NUMINAMATH_CALUDE_center_is_five_l1858_185891


namespace NUMINAMATH_CALUDE_propositions_true_l1858_185887

theorem propositions_true (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 - b^2 = 1 → a - b < 1) ∧
  (Real.exp a - Real.exp b = 1 → a - b < 1) := by
  sorry

end NUMINAMATH_CALUDE_propositions_true_l1858_185887


namespace NUMINAMATH_CALUDE_raisin_cost_fraction_l1858_185862

/-- Represents the cost ratio of raisins to the total mixture -/
def cost_ratio (raisin_weight : ℚ) (nut_weight : ℚ) (nut_cost_ratio : ℚ) : ℚ :=
  (raisin_weight) / (raisin_weight + nut_weight * nut_cost_ratio)

/-- Theorem stating that the cost of raisins is 3/19 of the total mixture cost -/
theorem raisin_cost_fraction :
  cost_ratio 3 4 4 = 3 / 19 := by
sorry

end NUMINAMATH_CALUDE_raisin_cost_fraction_l1858_185862


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l1858_185853

theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples : ℝ) * 0.4 = 300 → initial_apples = 750 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l1858_185853


namespace NUMINAMATH_CALUDE_triangle_area_proof_l1858_185877

theorem triangle_area_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  b * Real.cos A = (Real.sqrt 2 * c - a) * Real.cos B →
  B = π / 4 →
  C > π / 2 →
  a = 4 →
  b = 3 →
  (1 / 2) * a * b * Real.sin C = 4 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l1858_185877


namespace NUMINAMATH_CALUDE_geometric_parallelism_l1858_185825

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the "contained in" relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (line_plane_parallel : Line → Plane → Prop)

-- State the theorem
theorem geometric_parallelism 
  (a : Line) (α β : Plane) (h : contained_in a α) :
  (plane_parallel α β → line_plane_parallel a β) ∧
  (¬ line_plane_parallel a β → ¬ plane_parallel α β) ∧
  ¬ (line_plane_parallel a β → plane_parallel α β) :=
sorry

end NUMINAMATH_CALUDE_geometric_parallelism_l1858_185825


namespace NUMINAMATH_CALUDE_fish_tank_weeks_l1858_185818

/-- Represents the fish tank scenario -/
structure FishTank where
  initialTotal : ℕ
  dailyKoiAdded : ℕ
  dailyGoldfishAdded : ℕ
  finalKoi : ℕ
  finalGoldfish : ℕ

/-- Calculates the number of weeks fish were added to the tank -/
def weeksAdded (tank : FishTank) : ℚ :=
  let totalAdded := tank.finalKoi + tank.finalGoldfish - tank.initialTotal
  let dailyAdded := tank.dailyKoiAdded + tank.dailyGoldfishAdded
  (totalAdded : ℚ) / (dailyAdded * 7 : ℚ)

/-- Theorem stating that for the given scenario, fish were added for 3 weeks -/
theorem fish_tank_weeks (tank : FishTank) 
  (h1 : tank.initialTotal = 280)
  (h2 : tank.dailyKoiAdded = 2)
  (h3 : tank.dailyGoldfishAdded = 5)
  (h4 : tank.finalKoi = 227)
  (h5 : tank.finalGoldfish = 200) :
  weeksAdded tank = 3 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_weeks_l1858_185818


namespace NUMINAMATH_CALUDE_inverse_proportion_point_l1858_185837

theorem inverse_proportion_point : 
  let x : ℝ := 2 * Real.sqrt 2
  let y : ℝ := Real.sqrt 2
  y = 4 / x := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_l1858_185837


namespace NUMINAMATH_CALUDE_photo_archive_album_size_l1858_185852

/-- Represents an album in the photo archive -/
structure Album where
  pages : ℕ
  photos_per_page : ℕ

/-- The photo archive system -/
structure PhotoArchive where
  album : Album
  /-- Ensures all albums are identical -/
  albums_identical : ∀ a b : Album, a = b

theorem photo_archive_album_size 
  (archive : PhotoArchive)
  (h1 : archive.album.photos_per_page = 4)
  (h2 : ∃ x : ℕ, 81 = (x - 1) * (archive.album.pages * archive.album.photos_per_page) + 5 * archive.album.photos_per_page)
  (h3 : ∃ y : ℕ, 171 = (y - 1) * (archive.album.pages * archive.album.photos_per_page) + 3 * archive.album.photos_per_page)
  : archive.album.pages * archive.album.photos_per_page = 32 := by
  sorry

end NUMINAMATH_CALUDE_photo_archive_album_size_l1858_185852


namespace NUMINAMATH_CALUDE_orchard_tree_difference_l1858_185831

theorem orchard_tree_difference : 
  let ahmed_orange : ℕ := 8
  let hassan_apple : ℕ := 1
  let hassan_orange : ℕ := 2
  let ahmed_apple : ℕ := 4 * hassan_apple
  let ahmed_total : ℕ := ahmed_orange + ahmed_apple
  let hassan_total : ℕ := hassan_apple + hassan_orange
  ahmed_total - hassan_total = 9 := by
sorry

end NUMINAMATH_CALUDE_orchard_tree_difference_l1858_185831


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1858_185813

theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (1 / m + 1 / n) ≥ 4 ∧ ∃ m n, m > 0 ∧ n > 0 ∧ m + n = 1 ∧ 1 / m + 1 / n = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1858_185813


namespace NUMINAMATH_CALUDE_books_combination_l1858_185876

def choose (n : ℕ) (r : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

theorem books_combination : choose 15 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_books_combination_l1858_185876


namespace NUMINAMATH_CALUDE_closest_approximation_l1858_185857

def x_values : List ℝ := [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

def y (x : ℝ) : ℝ := x^2 - x

def distance_to_target (x : ℝ) : ℝ := |y x - 1.4|

theorem closest_approximation :
  ∀ x ∈ x_values, distance_to_target 1.8 ≤ distance_to_target x := by
  sorry

end NUMINAMATH_CALUDE_closest_approximation_l1858_185857


namespace NUMINAMATH_CALUDE_f_36_equals_2pq_l1858_185871

/-- A function satisfying f(xy) = f(x) + f(y) for all x and y -/
def LogLikeFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x * y) = f x + f y

/-- Main theorem: f(36) = 2(p + q) given the conditions -/
theorem f_36_equals_2pq (f : ℝ → ℝ) (p q : ℝ) 
  (h1 : LogLikeFunction f) 
  (h2 : f 2 = p) 
  (h3 : f 3 = q) : 
  f 36 = 2 * (p + q) := by
  sorry


end NUMINAMATH_CALUDE_f_36_equals_2pq_l1858_185871


namespace NUMINAMATH_CALUDE_log_sum_equals_one_implies_product_equals_ten_l1858_185886

theorem log_sum_equals_one_implies_product_equals_ten (a b : ℝ) (h : Real.log a + Real.log b = 1) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_one_implies_product_equals_ten_l1858_185886
