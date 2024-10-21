import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_addition_and_scalar_multiplication_l704_70417

theorem vector_addition_and_scalar_multiplication :
  let v₁ : Fin 3 → ℝ := ![(-3 : ℝ), 2, -5]
  let v₂ : Fin 3 → ℝ := ![1, 7, -3]
  (2 : ℝ) • (v₁ + v₂) = ![(-4 : ℝ), 18, -16] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_addition_and_scalar_multiplication_l704_70417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l704_70478

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, 
    prove the perimeter and cos(A-C) given specific side lengths and cos C -/
theorem triangle_properties (a b c : ℝ) (A B C : Real) :
  a = 1 →
  b = 2 →
  Real.cos C = 1/4 →
  a + b + c = 5 ∧
  Real.cos (A - C) = 11/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l704_70478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l704_70483

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  (sum_n seq 3 = 12) →
  (∃ r : ℚ, seq.a 2 = r * (2 * seq.a 1) ∧ seq.a 3 + 1 = r * seq.a 2) →
  (∀ n : ℕ, sum_n seq n = 1/2 * n * (3*n - 1) ∨ sum_n seq n = 2*n * (5 - n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l704_70483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l704_70412

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 4 * x^2 + 1/x

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y : ℝ, 1/2 < x ∧ x < y → f x < f y :=
by
  -- Introduce variables and hypothesis
  intro x y h
  -- Extract the inequalities from the hypothesis
  have hx : 1/2 < x := h.left
  have hxy : x < y := h.right
  -- Apply the definition of f
  unfold f
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l704_70412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unbounded_growth_l704_70472

/-- Represents the state of the four integers at any iteration -/
structure State where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- The iteration function that transforms the state -/
def iterate (s : State) : State :=
  { a := s.a - s.b,
    b := s.b - s.c,
    c := s.c - s.d,
    d := s.d - s.a }

/-- Predicate to check if all integers in a state are equal -/
def allEqual (s : State) : Prop :=
  s.a = s.b ∧ s.b = s.c ∧ s.c = s.d

/-- The main theorem statement -/
theorem unbounded_growth (initial : State) (h : ¬allEqual initial) :
  ∃ (f : ℕ → ℤ), ∀ (M : ℕ), ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N →
    (let s := (iterate^[n] initial)
     |s.a| ≥ M ∨ |s.b| ≥ M ∨ |s.c| ≥ M ∨ |s.d| ≥ M) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unbounded_growth_l704_70472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_increase_l704_70430

noncomputable def next_side_length (current_length : ℝ) (percentage : ℝ) : ℝ :=
  current_length * percentage / 100

noncomputable def triangle_perimeter (side_length : ℝ) : ℝ :=
  3 * side_length

noncomputable def percent_increase (initial : ℝ) (final : ℝ) : ℝ :=
  (final - initial) / initial * 100

theorem triangle_perimeter_increase :
  let s₁ : ℝ := 4
  let s₂ : ℝ := next_side_length s₁ 150
  let s₃ : ℝ := next_side_length s₂ 130
  let s₄ : ℝ := next_side_length s₃ 150
  let s₅ : ℝ := next_side_length s₄ 130
  let p₁ : ℝ := triangle_perimeter s₁
  let p₅ : ℝ := triangle_perimeter s₅
  abs (percent_increase p₁ p₅ - 280.3) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_increase_l704_70430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_properties_l704_70482

-- Define the average operation
noncomputable def avg (x y : ℝ) : ℝ := (3 * x + 2 * y) / 5

-- State the theorem
theorem avg_properties :
  -- Multiplication distributes over avg
  (∀ x y z : ℝ, x * avg y z = avg (x * y) (x * z)) ∧
  -- avg is not associative
  (∃ x y z : ℝ, avg (avg x y) z ≠ avg x (avg y z)) ∧
  -- avg is not commutative
  (∃ x y : ℝ, avg x y ≠ avg y x) ∧
  -- avg does not distribute over multiplication
  (∃ x y z : ℝ, avg x (y * z) ≠ avg x y * avg x z) ∧
  -- avg does not have an identity element
  (¬ ∃ e : ℝ, ∀ x : ℝ, avg x e = x ∧ avg e x = x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_properties_l704_70482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l704_70420

theorem right_triangle_area (b h A : ℝ) : 
  b > 0 →  -- Ensure positive base
  h > b →  -- Given condition: b < h
  h = (5/4) * b →  -- Ratio of hypotenuse to base is 5:4
  A = (1/2) * b * (Real.sqrt ((h^2 - b^2) / b^2)) * b →  -- Area formula using height from Pythagorean theorem
  A = (3/8) * b^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l704_70420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_valid_number_is_744_l704_70407

def is_valid (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 800

def random_sequence : List Nat :=
  [844, 217, 533, 157, 425, 506, 887, 704, 744]

theorem seventh_valid_number_is_744 :
  (random_sequence.filter is_valid).get? 6 = some 744 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_valid_number_is_744_l704_70407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_determinant_zero_l704_70499

theorem system_determinant_zero (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) :
  let k : ℝ := 103 / 13
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![1, k, 4; 4, k, -3; 3, 5, -4]
  x + k * y + 4 * z = 0 ∧ 4 * x + k * y - 3 * z = 0 ∧ 3 * x + 5 * y - 4 * z = 0 →
  Matrix.det A = 0 := by
  intros k A h
  sorry

#check system_determinant_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_determinant_zero_l704_70499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_game_strategy_l704_70442

/-- Represents the state of the candy game -/
structure GameState where
  candies : Nat
  isFirstPlayerTurn : Bool

/-- Checks if a move is valid in the candy game -/
def isValidMove (state : GameState) (move : Nat) : Prop :=
  1 ≤ move ∧ move ≤ state.candies ∧ Nat.Coprime move state.candies

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Nat) : GameState :=
  { candies := state.candies - move,
    isFirstPlayerTurn := ¬state.isFirstPlayerTurn }

/-- Determines if the current state is a winning position for the current player -/
def isWinningPosition : GameState → Prop
| state => state.candies % 2 = 1

/-- Theorem stating the winning strategy for the candy game -/
theorem candy_game_strategy (n : Nat) :
  let initialState : GameState := { candies := n, isFirstPlayerTurn := true }
  isWinningPosition initialState ↔ n % 2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_game_strategy_l704_70442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_correct_l704_70415

def mySequence (n : ℕ) : ℤ := (-1)^(n+1) * (4*n - 1)

theorem mySequence_correct : ∀ n : ℕ, n ≥ 1 →
  (mySequence n = 3 ∧ n = 1) ∨
  (mySequence n = -7 ∧ n = 2) ∨
  (mySequence n = 11 ∧ n = 3) ∨
  (mySequence n = -15 ∧ n = 4) ∨
  (mySequence (n+1) = -mySequence n + 4) :=
by
  sorry

#eval mySequence 1
#eval mySequence 2
#eval mySequence 3
#eval mySequence 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_correct_l704_70415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motel_rent_theorem_l704_70474

/-- Represents the total rent charged by the motel -/
def total_rent (x y : ℕ) : ℕ := 40 * x + 60 * y

/-- Represents the total rent if 10 rooms were moved from $60 to $40 -/
def adjusted_rent (x y : ℕ) : ℕ := 40 * (x + 10) + 60 * (y - 10)

theorem motel_rent_theorem (x y : ℕ) :
  adjusted_rent x y = (90 * total_rent x y) / 100 →
  total_rent x y = 2000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_motel_rent_theorem_l704_70474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_petya_advantage_l704_70471

/-- Represents the vote counts for a candidate over two time periods -/
structure VoteCounts where
  first_period : ℕ
  second_period : ℕ

/-- The election results for the class president -/
structure ElectionResult where
  petya : VoteCounts
  vasya : VoteCounts

def total_votes (result : ElectionResult) : ℕ :=
  result.petya.first_period + result.petya.second_period +
  result.vasya.first_period + result.vasya.second_period

def petya_advantage (result : ElectionResult) : ℤ :=
  (result.petya.first_period + result.petya.second_period : ℤ) -
  (result.vasya.first_period + result.vasya.second_period : ℤ)

theorem max_petya_advantage (result : ElectionResult) : 
  total_votes result = 27 →
  result.petya.first_period = result.vasya.first_period + 9 →
  result.vasya.second_period = result.petya.second_period + 9 →
  petya_advantage result > 0 →
  petya_advantage result ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_petya_advantage_l704_70471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_A_for_F_composition_l704_70495

/-- Definition of F_n(a) -/
def F (n a : ℕ) : ℕ := 
  let q := a / n
  let r := a % n
  q + r

/-- Theorem statement -/
theorem largest_A_for_F_composition : 
  ∃ (n₁ n₂ n₃ n₄ n₅ n₆ : ℕ), 
    (∀ a : ℕ, 0 < a ∧ a ≤ 53590 → 
      F n₆ (F n₅ (F n₄ (F n₃ (F n₂ (F n₁ a))))) = 1) ∧
    (∀ A : ℕ, A > 53590 → 
      ¬∃ (m₁ m₂ m₃ m₄ m₅ m₆ : ℕ), 
        ∀ a : ℕ, 0 < a ∧ a ≤ A → 
          F m₆ (F m₅ (F m₄ (F m₃ (F m₂ (F m₁ a))))) = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_A_for_F_composition_l704_70495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_implies_a_eq_two_non_negative_implies_a_in_zero_one_l704_70445

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) - a * x / (x + 1)

-- Theorem 1: If x = 1 is an extremum point, then a = 2
theorem extremum_point_implies_a_eq_two (a : ℝ) (h : a > 0) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1 ∨ f a x ≥ f a 1) →
  a = 2 := by
  sorry

-- Theorem 2: If f(x) ≥ 0 for all x ≥ 0, then 0 < a ≤ 1
theorem non_negative_implies_a_in_zero_one (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ 0) →
  a ∈ Set.Ioc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_implies_a_eq_two_non_negative_implies_a_in_zero_one_l704_70445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_is_126_l704_70484

/-- Represents the capacity of a water tank in gallons. -/
def tank_capacity (c : ℝ) : Prop := c > 0

/-- The difference in gallons between 90% and 40% full is 63 gallons. -/
axiom volume_difference (c : ℝ) : tank_capacity c → 0.9 * c - 0.4 * c = 63

theorem tank_capacity_is_126 : 
  ∃ c : ℝ, tank_capacity c ∧ c = 126 :=
by
  -- We'll use 126 as our witness
  use 126
  constructor
  · -- Prove that 126 satisfies tank_capacity
    unfold tank_capacity
    norm_num
  · -- Prove that 126 satisfies the volume difference equation
    rfl

#check tank_capacity_is_126

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_is_126_l704_70484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_covered_is_eight_l704_70449

/-- Represents a square card -/
structure Card where
  side : ℝ

/-- Represents a square on the checkerboard -/
structure Square where
  side : ℝ

/-- Represents the number of squares covered by the card -/
def squaresCovered (card : Card) (square : Square) : ℕ :=
  sorry

/-- The maximum number of squares that can be covered by the card -/
def maxSquaresCovered (card : Card) (square : Square) : ℕ := 
  max (squaresCovered card square) (squaresCovered card square)

theorem max_squares_covered_is_eight :
  ∀ (card : Card) (square : Square),
    card.side = 2 ∧ square.side = 1 →
    maxSquaresCovered card square = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_covered_is_eight_l704_70449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l704_70485

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- hyperbola equation
  (∃ M : ℝ × ℝ, 
    M.1 = c ∧ M.2 = b^2 / a ∧  -- M is on the right branch
    M.2 / (M.1 + c) = 1 / Real.sqrt 3 ∧  -- slope of F₁M is tan 30°
    M.2 / (M.1 - c) = 0) →  -- MF₂ is perpendicular to x-axis
  c / a = Real.sqrt 3 :=  -- eccentricity is √3
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l704_70485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_segment_probability_l704_70491

/-- Represents the set of all sides and diagonals in a regular hexagon -/
inductive HexagonSegments
  | Side
  | ShortDiagonal
  | LongDiagonal

/-- Classifies a segment as a side, short diagonal, or long diagonal -/
def segment_type (s : HexagonSegments) : Fin 3 :=
  match s with
  | HexagonSegments.Side => 0
  | HexagonSegments.ShortDiagonal => 1
  | HexagonSegments.LongDiagonal => 2

/-- The number of segments of each type -/
def segment_count : Fin 3 → Nat
  | 0 => 6  -- sides
  | 1 => 6  -- short diagonals
  | 2 => 3  -- long diagonals
  | _ => 0  -- This case should never occur, but Lean requires it for exhaustiveness

/-- The probability of selecting two segments of the same length -/
def same_length_probability : ℚ := 11 / 35

theorem hexagon_segment_probability :
  let total : Nat := 15
  let prob_side := (segment_count 0 : ℚ) / total
  let prob_short_diag := (segment_count 1 : ℚ) / total
  let prob_long_diag := (segment_count 2 : ℚ) / total
  (prob_side * ((segment_count 0 - 1) : ℚ) / (total - 1) +
   prob_short_diag * ((segment_count 1 - 1) : ℚ) / (total - 1) +
   prob_long_diag * ((segment_count 2 - 1) : ℚ) / (total - 1)) = same_length_probability :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_segment_probability_l704_70491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_equals_interval_l704_70439

-- Define the set A
def A : Set ℝ := {y | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem set_A_equals_interval : A = Set.Icc (-4) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_equals_interval_l704_70439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bisections_for_accuracy_l704_70447

/-- Given a function f with a root in (1,2), proves the minimum number of bisections
    required to achieve an accuracy of 0.01 -/
theorem min_bisections_for_accuracy (f : ℝ → ℝ) : 
  (∃ x : ℝ, x > 1 ∧ x < 2 ∧ f x = 0) →  -- f has a root in (1,2)
  (∀ n : ℕ, ((1 : ℝ) / 2^n ≤ 0.01 → n ≥ 7) ∧
            (n ≥ 7 → (1 : ℝ) / 2^n ≤ 0.01)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bisections_for_accuracy_l704_70447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l704_70436

-- Define the lines
noncomputable def line1 (x : ℝ) : ℝ := x + 2
noncomputable def line2 (x : ℝ) : ℝ := -3 * x + 9
noncomputable def line3 : ℝ := 2

-- Define the vertices of the triangle
noncomputable def vertex1 : ℝ × ℝ := (0, 2)
noncomputable def vertex2 : ℝ × ℝ := (7/3, 2)
noncomputable def vertex3 : ℝ × ℝ := (7/4, 15/4)

-- Theorem statement
theorem triangle_area : 
  let base := vertex2.1 - vertex1.1
  let height := vertex3.2 - vertex1.2
  (1/2 : ℝ) * base * height = 49/24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l704_70436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l₁_l₂_properties_l704_70421

/-- Line l₁ defined by x - y - 1 = 0 -/
def l₁ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 - 1 = 0}

/-- Line l₂ defined by (k + 1)x + ky + k = 0, parameterized by k -/
def l₂ (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (k + 1) * p.1 + k * p.2 + k = 0}

/-- The slope of l₂ for a given k -/
noncomputable def slope_l₂ (k : ℝ) : ℝ := -(k + 1) / k

/-- Theorem stating the three properties of l₁ and l₂ -/
theorem l₁_l₂_properties :
  (∃ k : ℝ, k = 0) ∧ 
  (∀ k : ℝ, ∃ p : ℝ × ℝ, p ∈ l₁ ∧ p ∈ l₂ k) ∧
  (∀ k : ℝ, slope_l₂ k ≠ -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_l₁_l₂_properties_l704_70421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l704_70454

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos (2 * x)

theorem function_properties :
  ∃ (α : ℝ), 0 < α ∧ α < π / 3 ∧ f α = 6 / 5 ∧
  (∀ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) → T ≥ π) ∧
  Real.sin (2 * α) = (3 * Real.sqrt 3 + 4) / 10 := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l704_70454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_value_l704_70461

open Real

theorem trigonometric_expression_value (θ : ℝ) 
  (h : Real.cos (θ + π / 2) = -1 / 2) : 
  (Real.cos (θ + π) / (Real.sin (π / 2 - θ) * (Real.cos (3 * π - θ) - 1))) + 
  (Real.cos (θ - 2 * π) / (Real.cos (-θ) * Real.cos (π - θ) + Real.sin (θ + 5 * π / 2))) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_value_l704_70461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AP_l704_70451

/-- Quadrilateral type -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- Rectangle type -/
structure Rectangle where
  topLeft : ℝ × ℝ
  bottomRight : ℝ × ℝ

/-- Check if a quadrilateral is a square -/
def Quadrilateral.isSquare (q : Quadrilateral) : Prop :=
  sorry

/-- Get the side length of a square -/
def Quadrilateral.sideLength (q : Quadrilateral) : ℝ :=
  sorry

/-- Get a specific side of a quadrilateral -/
def Quadrilateral.sideAD (q : Quadrilateral) : ℝ × ℝ :=
  sorry

/-- Get a specific side of a rectangle -/
def Rectangle.sideWX (r : Rectangle) : ℝ × ℝ :=
  sorry

/-- Calculate the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ :=
  sorry

/-- Calculate the shaded area given a quadrilateral and a rectangle -/
def shadedArea (q : Quadrilateral) (r : Rectangle) : ℝ :=
  sorry

/-- Calculate the length of AP given a quadrilateral -/
def lengthAP (q : Quadrilateral) : ℝ :=
  sorry

/-- Perpendicularity of lines -/
def Perpendicular (l1 l2 : ℝ × ℝ) : Prop :=
  sorry

/-- Given a square ABCD and a rectangle WXYZ with specific properties, 
    prove that the length of AP is 4. -/
theorem length_of_AP (ABCD : Quadrilateral) (WXYZ : Rectangle) 
  (h1 : ABCD.isSquare)
  (h2 : ABCD.sideLength = 8)
  (h3 : WXYZ.bottomRight.1 - WXYZ.topLeft.1 = 12)  -- length
  (h4 : WXYZ.topLeft.2 - WXYZ.bottomRight.2 = 8)   -- width
  (h5 : Perpendicular (ABCD.sideAD) (WXYZ.sideWX))
  (h6 : shadedArea ABCD WXYZ = (1/3) * WXYZ.area) :
  lengthAP ABCD = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AP_l704_70451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_quadrilateral_area_l704_70437

/-- The area of a quadrilateral given its diagonal and two offsets -/
noncomputable def quadrilateralArea (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) : ℝ :=
  (1 / 2) * diagonal * (offset1 + offset2)

/-- Theorem: The area of a quadrilateral with diagonal 10 cm and offsets 7 cm and 3 cm is 50 cm² -/
theorem specific_quadrilateral_area :
  quadrilateralArea 10 7 3 = 50 := by
  -- Unfold the definition of quadrilateralArea
  unfold quadrilateralArea
  -- Simplify the arithmetic
  simp [mul_add, mul_assoc]
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_quadrilateral_area_l704_70437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_barons_in_kingdom_l704_70431

-- Define the Knight type
def Knight : Type := ℕ

-- Define the relation of vassalage
def isVassal : Knight → Knight → Prop := sorry

-- Define the wealth relation
def isWealthier : Knight → Knight → Prop := sorry

-- Define what it means to be a baron
def isBaron (k : Knight) : Prop :=
  ∃ (vassals : Finset Knight), vassals.card ≥ 4 ∧ ∀ v, v ∈ vassals → isVassal v k

-- State the theorem
theorem max_barons_in_kingdom :
  ∀ (knights : Finset Knight),
    knights.card = 32 →
    (∀ k₁ k₂ k₃, k₁ ∈ knights → k₂ ∈ knights → k₃ ∈ knights → 
      isVassal k₁ k₂ → isVassal k₂ k₃ → ¬isVassal k₁ k₃) →
    (∀ k₁ k₂, k₁ ∈ knights → k₂ ∈ knights → isVassal k₁ k₂ → isWealthier k₂ k₁) →
    (∀ k, k ∈ knights → ∃! l, l ∈ knights ∧ (isVassal k l ∨ k = l)) →
    (∀ (barons : Finset Knight),
      (∀ b, b ∈ barons → b ∈ knights ∧ isBaron b) →
      barons.card ≤ 7) ∧
    ∃ (barons : Finset Knight),
      (∀ b, b ∈ barons → b ∈ knights ∧ isBaron b) ∧
      barons.card = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_barons_in_kingdom_l704_70431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l704_70475

/-- The point P in the problem -/
def P : ℝ × ℝ := (2, 3)

/-- The function representing the family of lines ax + (a-1)y + 3 = 0 -/
def line (a : ℝ) (x y : ℝ) : ℝ := a * x + (a - 1) * y + 3

/-- The distance function between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The distance from a point to a line given by ax + by + c = 0 -/
noncomputable def distanceToLine (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / Real.sqrt (a^2 + b^2)

theorem max_distance_to_line :
  ∃ (d : ℝ) (a : ℝ),
    d = 5 ∧
    a = 1 ∧
    ∀ (a' : ℝ),
      distanceToLine P a' (a' - 1) 3 ≤ d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l704_70475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l704_70411

open Real

-- Define the polar equation of the line
def line (ρ θ : ℝ) : Prop := ρ * cos θ - ρ * sin θ - 1 = 0

-- Define the parametric equations of the ellipse
def ellipse (x y θ : ℝ) : Prop := x = 2 * cos θ ∧ y = sin θ

-- Define the point P where the line intersects the x-axis
def P : ℝ × ℝ := (1, 0)

-- Define the points A and B where the line intersects the ellipse
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ (θ₁ θ₂ ρ₁ ρ₂ : ℝ),
    line ρ₁ θ₁ ∧ line ρ₂ θ₂ ∧
    ellipse A.1 A.2 θ₁ ∧ ellipse B.1 B.2 θ₂

-- State the theorem
theorem intersection_product (A B : ℝ × ℝ) :
  intersection_points A B →
  (A.1 - P.1)^2 + (A.2 - P.2)^2 * ((B.1 - P.1)^2 + (B.2 - P.2)^2) = (6/5)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l704_70411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_interior_angle_mean_l704_70434

/-- A quadrilateral is a polygon with four sides and four angles. -/
structure Quadrilateral where

/-- The sum of interior angles in a quadrilateral is 360°. -/
def sum_of_interior_angles (q : Quadrilateral) : ℝ := 360

/-- The number of interior angles in a quadrilateral is 4. -/
def number_of_interior_angles (q : Quadrilateral) : ℝ := 4

/-- The mean value of the measures of the four interior angles of any quadrilateral is 90°. -/
theorem quadrilateral_interior_angle_mean :
  ∀ (q : Quadrilateral), 
  (sum_of_interior_angles q) / (number_of_interior_angles q) = 90 :=
by
  intro q
  simp [sum_of_interior_angles, number_of_interior_angles]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_interior_angle_mean_l704_70434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_y_coordinate_l704_70452

-- Define the points
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (8, 12)
def C : ℝ × ℝ := (14, 0)
def X : ℝ × ℝ := (6, 0)
def Y : ℝ × ℝ := (8, 4)

-- Define the area ratio
def area_ratio : ℝ := 0.1111111111111111

-- Function to calculate the area of a triangle given three points
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

-- Theorem statement
theorem z_y_coordinate : 
  ∃ (z : ℝ × ℝ), 
    let (zx, zy) := z
    triangle_area X Y z = area_ratio * triangle_area A B C ∧ 
    zy = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_y_coordinate_l704_70452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_log_quadratic_l704_70497

/-- The function f(x) = log_3(x^2 - 2x - 3) is monotonically increasing on the interval (3, +∞) -/
theorem monotonic_increasing_log_quadratic :
  ∀ x y, 3 < x → x < y → (Real.log (x^2 - 2*x - 3) < Real.log (y^2 - 2*y - 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_log_quadratic_l704_70497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_concentration_example_l704_70496

/-- Calculates the salt concentration in a mixture of pure water and salt solution -/
noncomputable def salt_concentration (pure_water_volume : ℝ) (salt_solution_volume : ℝ) (salt_solution_concentration : ℝ) : ℝ :=
  let total_volume := pure_water_volume + salt_solution_volume
  let salt_amount := salt_solution_volume * salt_solution_concentration
  salt_amount / total_volume

/-- Theorem: The salt concentration in a mixture of 1 liter pure water and 0.25 liters of 75% salt solution is 15% -/
theorem salt_concentration_example : salt_concentration 1 0.25 0.75 = 0.15 := by
  unfold salt_concentration
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_concentration_example_l704_70496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l704_70453

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2*x + 2 - 2*Real.sqrt (2*x + 1)) + Real.sqrt (2*x + 10 - 6*Real.sqrt (2*x + 1))

theorem solution_set (x : ℝ) : f x = 2 ↔ x ∈ Set.Icc 0 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l704_70453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_stability_l704_70424

/-- Represents the state of a wise man -/
inductive Opinion : Type
| Earth : Opinion  -- Earth revolves around Jupiter
| Jupiter : Opinion  -- Jupiter revolves around Earth

instance : DecidableEq Opinion :=
  fun a b =>
    match a, b with
    | Opinion.Earth, Opinion.Earth => isTrue rfl
    | Opinion.Jupiter, Opinion.Jupiter => isTrue rfl
    | Opinion.Earth, Opinion.Jupiter => isFalse (fun h => Opinion.noConfusion h)
    | Opinion.Jupiter, Opinion.Earth => isFalse (fun h => Opinion.noConfusion h)

/-- Represents the circle of wise men -/
def Circle := Fin 101 → Opinion

/-- Function to update the opinion of a wise man based on his neighbors -/
def updateOpinion (left right current : Opinion) : Opinion :=
  if left ≠ current ∧ right ≠ current then
    match current with
    | Opinion.Earth => Opinion.Jupiter
    | Opinion.Jupiter => Opinion.Earth
  else
    current

/-- Function to update the entire circle of opinions -/
def updateCircle (c : Circle) : Circle :=
  fun i =>
    updateOpinion (c (i - 1)) (c (i + 1)) (c i)

/-- Predicate to check if a circle is stable (no changes occur) -/
def isStable (c : Circle) : Prop :=
  c = updateCircle c

/-- Main theorem: The system will eventually reach a stable configuration -/
theorem eventual_stability (initial : Circle) :
  ∃ n : ℕ, isStable (n.iterate updateCircle initial) := by
  sorry

#check eventual_stability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_stability_l704_70424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_and_uniqueness_l704_70467

/-- Given the length of a side, the measure of the opposite angle, and the length of a median,
    there exists a unique triangle satisfying these conditions. -/
theorem triangle_existence_and_uniqueness 
  (a : ℝ) (α : ℝ) (m : ℝ) 
  (h_a : a > 0) (h_α : 0 < α ∧ α < π) (h_m : m > 0) : 
  ∃! (A B C : ℝ × ℝ), 
    let d := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
    let θ := Real.arccos ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) / 
              (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * 
               Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))
    let midpoint := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
    let median_length := Real.sqrt ((A.1 - midpoint.1)^2 + (A.2 - midpoint.2)^2)
    d = a ∧ θ = α ∧ median_length = m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_and_uniqueness_l704_70467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l704_70418

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem sum_of_f_values :
  f 1 + f 2 + f 3 + f 4 + f (1/2) + f (1/3) + f (1/4) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l704_70418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_36_l704_70464

theorem number_of_divisors_36 : 
  Finset.card (Finset.filter (fun d => 36 % d = 0) (Finset.range 37)) = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_36_l704_70464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l704_70480

/-- An arithmetic sequence with a common difference less than zero -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  d_negative : d < 0

/-- The sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_max_sum (seq : ArithmeticSequence) 
    (h : S seq 8 = S seq 12) :
    ∃ (n : ℕ), (∀ (m : ℕ), S seq m ≤ S seq n) ∧ n = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l704_70480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_to_negative_half_power_l704_70404

theorem sixteen_to_negative_half_power (x : ℝ) : x = 16 → x^(-(1/2 : ℝ)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_to_negative_half_power_l704_70404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l704_70444

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 1

-- Define the point of tangency
def point : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a * x + b * y + c = 0 ↔ 
      y = ((deriv f) point.1) * (x - point.1) + f point.1) ∧
    a = 1 ∧ b = -1 ∧ c = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l704_70444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maxwell_brad_meeting_time_l704_70426

/-- The time it takes for Maxwell and Brad to meet after Brad starts running -/
noncomputable def meeting_time (distance : ℝ) (maxwell_speed : ℝ) (brad_speed : ℝ) : ℝ :=
  (distance - maxwell_speed) / (maxwell_speed + brad_speed)

/-- The total distance Maxwell walks before meeting Brad -/
def maxwell_distance (t : ℝ) (speed : ℝ) : ℝ := speed * (t + 1)

/-- The distance Brad runs before meeting Maxwell -/
def brad_distance (t : ℝ) (speed : ℝ) : ℝ := speed * t

theorem maxwell_brad_meeting_time :
  let distance := (94 : ℝ) -- km
  let maxwell_speed := (4 : ℝ) -- km/h
  let brad_speed := (6 : ℝ) -- km/h
  let t := meeting_time distance maxwell_speed brad_speed
  maxwell_distance t maxwell_speed + brad_distance t brad_speed = distance ∧
  t + 1 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maxwell_brad_meeting_time_l704_70426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l704_70400

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the configuration of circles C, D, and E -/
structure CircleConfiguration where
  C : Circle
  D : Circle
  E : Circle

/-- Predicate to check if circle A is internally tangent to circle B -/
def is_internally_tangent (A B : Circle) : Prop :=
  sorry

/-- Predicate to check if circle A is externally tangent to circle B -/
def is_externally_tangent (A B : Circle) : Prop :=
  sorry

/-- Predicate to check if circle A is tangent to a diameter of circle B -/
def is_tangent_to_diameter (A B : Circle) : Prop :=
  sorry

/-- The main theorem statement -/
theorem circle_tangency_theorem (config : CircleConfiguration) :
  config.C.radius = 2 ∧
  is_internally_tangent config.D config.C ∧
  is_internally_tangent config.E config.C ∧
  is_externally_tangent config.E config.D ∧
  is_tangent_to_diameter config.E config.C ∧
  config.D.radius = 3 * config.E.radius ∧
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ config.D.radius = Real.sqrt m - n ∧ m + n = 254 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l704_70400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painted_cube_count_l704_70489

/-- Represents a 3D coordinate in the block --/
structure Coord where
  x : Fin 5
  y : Fin 5
  z : Fin 1
deriving Fintype, Repr

/-- Counts the number of painted faces for a cube at a given coordinate --/
def countPaintedFaces (c : Coord) : Nat :=
  (if c.x = 0 ∨ c.x = 4 then 1 else 0) +
  (if c.y = 0 ∨ c.y = 4 then 1 else 0) +
  (if c.z = 0 then 1 else 0)

/-- Checks if a number is even --/
def isEven (n : Nat) : Bool :=
  n % 2 = 0

/-- The main theorem to prove --/
theorem painted_cube_count :
  (Finset.univ.filter (fun c => isEven (countPaintedFaces c))).card = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_painted_cube_count_l704_70489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_and_variance_of_transformed_data_l704_70443

noncomputable section

variable (x : Fin 5 → ℝ)

def average (data : Fin 5 → ℝ) : ℝ := (data 0 + data 1 + data 2 + data 3 + data 4) / 5

def variance (data : Fin 5 → ℝ) : ℝ :=
  ((data 0 - average data)^2 + (data 1 - average data)^2 + (data 2 - average data)^2 +
   (data 3 - average data)^2 + (data 4 - average data)^2) / 5

def transformed_data (data : Fin 5 → ℝ) : Fin 5 → ℝ :=
  fun i => 3 * data i - 2

theorem average_and_variance_of_transformed_data
  (h_avg : average x = 2)
  (h_var : variance x = 1/3) :
  average (transformed_data x) = 4 ∧ variance (transformed_data x) = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_and_variance_of_transformed_data_l704_70443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_triangle_areas_l704_70481

noncomputable section

-- Define necessary helper functions and structures
def Point : Type := ℝ × ℝ

def IsChord (A B : Point) (c : Set Point) : Prop := sorry
def Intersect (s1 s2 : Set Point) : Prop := sorry
def AngleMeasure (A B C : Point) : ℝ := sorry
def Segment (A B : Point) : Set Point := sorry
def Triangle (A B C : Point) : Set Point := sorry
def Area (s : Set Point) : ℝ := sorry

theorem chord_triangle_areas (R : ℝ) (A B C D M : Point) (circle : Set Point) :
  -- AB and CD are non-intersecting chords
  IsChord A B circle ∧ IsChord C D circle ∧ ¬ Intersect (Segment A B) (Segment C D) →
  -- ∠AB = 120°
  AngleMeasure A M B = 120 →
  -- ∠CD = 90°
  AngleMeasure C M D = 90 →
  -- M is the intersection point of chords AD and BC
  M ∈ Segment A D ∧ M ∈ Segment B C →
  -- The sum of areas of triangles AMB and CMD equals 100
  Area (Triangle A M B) + Area (Triangle C M D) = 100 →
  -- Then the area of triangle AMB is 60 and the area of triangle CMD is 40
  Area (Triangle A M B) = 60 ∧ Area (Triangle C M D) = 40 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_triangle_areas_l704_70481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_squares_l704_70435

/-- A point with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- A square with integer coordinates -/
structure IntSquare where
  bottomLeft : IntPoint
  size : Nat

/-- Check if a point is within the bounded region -/
def isInRegion (p : IntPoint) : Prop :=
  p.y ≥ -1 ∧ p.y ≤ 3 * p.x ∧ p.x ≤ 6 ∧ p.x ≥ 0

/-- Check if a square is entirely within the bounded region -/
def squareInRegion (s : IntSquare) : Prop :=
  isInRegion s.bottomLeft ∧
  isInRegion ⟨s.bottomLeft.x + s.size, s.bottomLeft.y⟩ ∧
  isInRegion ⟨s.bottomLeft.x, s.bottomLeft.y + s.size⟩ ∧
  isInRegion ⟨s.bottomLeft.x + s.size, s.bottomLeft.y + s.size⟩

/-- The set of all valid squares in the region -/
def validSquares : Set IntSquare :=
  {s | squareInRegion s ∧ (s.size = 1 ∨ s.size = 2 ∨ s.size = 4)}

/-- Theorem stating that the count of valid squares is 123 -/
theorem count_valid_squares : ∃ (s : Finset IntSquare), s.card = 123 ∧ ∀ sq, sq ∈ s ↔ sq ∈ validSquares := by
  sorry

#check count_valid_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_squares_l704_70435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_convexity_concavity_sin_sum_product_cos_sum_product_l704_70410

/-- Convexity and concavity of sine and cosine functions -/
theorem sin_cos_convexity_concavity :
  ∀ (k : ℤ) (x : ℝ),
  (((2 * k + 1 : ℝ) * Real.pi ≤ x ∧ x ≤ (2 * k + 2 : ℝ) * Real.pi) →
    ConvexOn ℝ (Set.Icc ((2 * k + 1 : ℝ) * Real.pi) ((2 * k + 2 : ℝ) * Real.pi)) Real.sin) ∧
  ((2 * k * Real.pi ≤ x ∧ x ≤ (2 * k + 1 : ℝ) * Real.pi) →
    ConcaveOn ℝ (Set.Icc (2 * k * Real.pi) ((2 * k + 1 : ℝ) * Real.pi)) Real.sin) ∧
  (((4 * k + 1 : ℝ) * Real.pi / 2 ≤ x ∧ x ≤ (4 * k + 3 : ℝ) * Real.pi / 2) →
    ConvexOn ℝ (Set.Icc ((4 * k + 1 : ℝ) * Real.pi / 2) ((4 * k + 3 : ℝ) * Real.pi / 2)) Real.cos) ∧
  (((4 * k - 1 : ℝ) * Real.pi / 2 ≤ x ∧ x ≤ (4 * k + 1 : ℝ) * Real.pi / 2) →
    ConcaveOn ℝ (Set.Icc ((4 * k - 1 : ℝ) * Real.pi / 2) ((4 * k + 1 : ℝ) * Real.pi / 2)) Real.cos) :=
by sorry

/-- Product form of sin(α + β) -/
theorem sin_sum_product (α β : ℝ) :
  Real.sin (α + β) = 2 * Real.sin ((α + β) / 2) * Real.cos ((α - β) / 2) :=
by sorry

/-- Product form of cos(α + β) -/
theorem cos_sum_product (α β : ℝ) :
  Real.cos (α + β) = 2 * Real.cos ((α + β) / 2) * Real.cos ((α - β) / 2) - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_convexity_concavity_sin_sum_product_cos_sum_product_l704_70410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_points_analysis_neg_two_local_max_four_local_min_l704_70479

/-- The function f(x) = x^3 + ax^2 + bx -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_points_analysis 
  (a b : ℝ) 
  (h1 : f' a b (-2) = 0)  -- x = -2 is an extremum point
  (h2 : f' a b 4 = 0)     -- x = 4 is an extremum point
  : 
  (a = -3 ∧ b = -24) ∧ 
  (∀ x < -2, f' (-3) (-24) x > 0) ∧ 
  (∀ x, -2 < x ∧ x < 4 → f' (-3) (-24) x < 0) ∧ 
  (∀ x > 4, f' (-3) (-24) x > 0) := by
  sorry

/-- x = -2 is a local maximum point -/
theorem neg_two_local_max 
  (a b : ℝ) 
  (h : a = -3 ∧ b = -24) :
  ∃ δ > 0, ∀ x, x ≠ -2 ∧ |x - (-2)| < δ → f a b x < f a b (-2) := by
  sorry

/-- x = 4 is a local minimum point -/
theorem four_local_min 
  (a b : ℝ) 
  (h : a = -3 ∧ b = -24) :
  ∃ δ > 0, ∀ x, x ≠ 4 ∧ |x - 4| < δ → f a b x > f a b 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_points_analysis_neg_two_local_max_four_local_min_l704_70479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_matrix_squared_is_identity_l704_70416

/-- Reflection matrix over a non-zero vector -/
def reflection_matrix (v : Fin 2 → ℝ) (h : v ≠ 0) : Matrix (Fin 2) (Fin 2) ℝ :=
  sorry

theorem reflection_matrix_squared_is_identity
  (v : Fin 2 → ℝ) (h : v ≠ 0) :
  (reflection_matrix v h) ^ 2 = Matrix.diagonal (λ _ => 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_matrix_squared_is_identity_l704_70416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_single_ticket_cost_is_18_l704_70428

noncomputable def total_sales : ℝ := 2280
def total_attendees : ℕ := 128
def couple_tickets_sold : ℕ := 16
noncomputable def couple_ticket_cost : ℝ := 35

noncomputable def single_ticket_cost : ℝ :=
  (total_sales - couple_tickets_sold * couple_ticket_cost) /
  (total_attendees - 2 * couple_tickets_sold)

theorem single_ticket_cost_is_18 :
  Int.floor single_ticket_cost = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_single_ticket_cost_is_18_l704_70428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l704_70487

theorem expression_simplification (k : ℤ) : 
  (3 : ℝ)^(-2*k) - (3 : ℝ)^(-(2*k-2)) + (3 : ℝ)^(-(2*k+1)) = -23/3 * (3 : ℝ)^(-2*k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l704_70487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_120_l704_70476

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_angle_120 (t : Triangle) :
  t.a^2 - t.b^2 = 3 * t.b * t.c ∧
  Real.sin t.C = 2 * Real.sin t.B →
  t.A = 120 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_120_l704_70476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_passes_through_fixed_point_a_range_for_inequality_infinitely_many_functions_between_l704_70413

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + Real.log x
def f₁ (x : ℝ) : ℝ := (1/6) * x^2 + (4/3) * x + (5/9) * Real.log x
def f₂ (x : ℝ) : ℝ := (1/2) * x^2 + 2 * a * x

theorem tangent_line_passes_through_fixed_point :
  ∃ (k m : ℝ), ∀ x, k * x + m = f a (Real.exp 1) + (deriv (f a)) (Real.exp 1) * (x - Real.exp 1) ∧
  k * (Real.exp 1 / 2) + m = 1/2 :=
sorry

theorem a_range_for_inequality :
  (∀ x > 1, f a x < f₂ a x) ↔ a ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ) :=
sorry

theorem infinitely_many_functions_between (h : a = 2/3) :
  ∃ g : ℝ → ℝ, ∀ x > 1, f₁ x < g x ∧ g x < f₂ (2/3) x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_passes_through_fixed_point_a_range_for_inequality_infinitely_many_functions_between_l704_70413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_statues_count_l704_70492

/-- Represents the number of turtle statues on Grandma Molly's lawn over four years --/
def TurtleStatues : ℕ → ℕ := sorry

/-- The number of statues broken in the third year --/
def BrokenStatues : ℕ := sorry

/-- Conditions from the problem --/
axiom initial_statues : TurtleStatues 1 = 4
axiom second_year : TurtleStatues 2 = 4 * TurtleStatues 1
axiom third_year_before_storm : TurtleStatues 2 + 12 = 28
axiom fourth_year_addition : TurtleStatues 4 - TurtleStatues 3 = 2 * BrokenStatues
axiom final_count : TurtleStatues 4 = 31

/-- Theorem stating that 3 statues were broken in the third year --/
theorem broken_statues_count : BrokenStatues = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_statues_count_l704_70492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l704_70441

-- Define P(n) as the greatest prime factor of n
noncomputable def P (n : ℕ) : ℕ := Nat.factors n |>.foldl max 1

theorem no_solution_exists : ¬∃ n : ℕ, n > 1 ∧ 
  (P n : ℝ) = Real.sqrt n ∧ 
  (P (n + 50) : ℝ) = Real.sqrt (n + 50) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l704_70441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_a_exists_l704_70488

theorem no_valid_a_exists : 
  ¬∃ (a : ℕ), 
    (0 < a ∧ a ≤ 100) ∧ 
    (∃ (x y : ℤ), x ≠ y ∧ 
      x^2 + (3*a + 1)*x + 2*a^2 = 0 ∧
      y^2 + (3*a + 1)*y + 2*a^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_a_exists_l704_70488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l704_70459

/-- The speed of the train in km/hr -/
noncomputable def train_speed : ℝ := 180

/-- The time taken by the train to cross the electric pole in seconds -/
noncomputable def crossing_time : ℝ := 3.499720022398208

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_hr_to_m_s : ℝ := 1000 / 3600

theorem train_length_calculation :
  let speed_m_s := train_speed * km_hr_to_m_s
  let train_length := speed_m_s * crossing_time
  ∃ ε > 0, |train_length - 174.986| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l704_70459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_l704_70486

/-- A plane intersecting the coordinate axes -/
structure IntersectingPlane where
  α : ℝ
  β : ℝ
  γ : ℝ
  h_distinct : α ≠ 0 ∧ β ≠ 0 ∧ γ ≠ 0
  h_distance : (1 / (2 * α))^2 + (1 / β)^2 + (1 / γ)^2 = 1 / 4

/-- The centroid of the triangle formed by the intersection points -/
noncomputable def centroid (plane : IntersectingPlane) : ℝ × ℝ × ℝ :=
  (2 * plane.α / 3, plane.β / 3, -plane.γ / 3)

/-- The theorem to be proved -/
theorem centroid_sum (plane : IntersectingPlane) :
  let (p, q, r) := centroid plane
  1 / p^2 + 1 / q^2 + 1 / r^2 = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_l704_70486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_x_equals_1_l704_70466

-- Define a vertical line
def vertical_line (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = a}

-- Define the angle of inclination for a vertical line
def angle_of_inclination_vertical : ℝ := 90

-- Define a general angle of inclination function (placeholder)
def angle_of_inclination (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem angle_of_inclination_x_equals_1 :
  angle_of_inclination_vertical = angle_of_inclination (vertical_line 1) :=
by
  sorry -- Proof to be filled in later

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_x_equals_1_l704_70466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterized_is_ellipse_parameterized_is_line_l704_70405

-- Define the parameter φ
variable (φ : ℝ)

-- Define the parameterized equations for the ellipse
noncomputable def x_ellipse (φ : ℝ) : ℝ := 5 * Real.cos φ
noncomputable def y_ellipse (φ : ℝ) : ℝ := 4 * Real.sin φ

-- Define the ellipse equation
def is_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Theorem for the ellipse
theorem parameterized_is_ellipse :
  ∀ φ, is_ellipse (x_ellipse φ) (y_ellipse φ) := by sorry

-- Define the parameter t
variable (t : ℝ)

-- Define the parameterized equations for the line
def x_line (t : ℝ) : ℝ := 1 - 3 * t
def y_line (t : ℝ) : ℝ := 4 * t

-- Define the line equation
def is_line (x y : ℝ) : Prop := 4 * x + 3 * y - 4 = 0

-- Theorem for the line
theorem parameterized_is_line :
  ∀ t, is_line (x_line t) (y_line t) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterized_is_ellipse_parameterized_is_line_l704_70405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_weight_is_328_l704_70423

/-- The ratio of weights of Meg's, Anne's, and Chris's cats -/
def weight_ratio : Fin 3 → ℕ
  | 0 => 13  -- Meg's cat
  | 1 => 21  -- Anne's cat
  | 2 => 28  -- Chris's cat

/-- The total weight of all three cats in kg -/
noncomputable def total_weight : ℝ := 496

/-- Meg's cat's weight in kg -/
noncomputable def meg_weight : ℝ := 20 + (weight_ratio 1 / weight_ratio 0) * total_weight / 2

/-- Anne's cat's weight in kg -/
noncomputable def anne_weight : ℝ := (weight_ratio 1 / weight_ratio 0) * meg_weight

/-- Chris's cat's weight in kg -/
noncomputable def chris_weight : ℝ := (weight_ratio 2 / weight_ratio 0) * meg_weight

/-- The combined weight of Meg's and Chris's cats in kg -/
noncomputable def combined_weight : ℝ := meg_weight + chris_weight

theorem combined_weight_is_328 : combined_weight = 328 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_weight_is_328_l704_70423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_fruit_juice_amount_correct_l704_70450

/-- The amount of mixed fruit juice opened to create a superfruit juice cocktail -/
noncomputable def mixed_fruit_juice_amount : ℝ :=
  42067 / 1136.6

/-- The cost of the superfruit juice cocktail per litre -/
def superfruit_cost_per_litre : ℝ := 1399.45

/-- The cost of the mixed fruit juice per litre -/
def mixed_fruit_cost_per_litre : ℝ := 262.85

/-- The cost of the açaí berry juice per litre -/
def acai_cost_per_litre : ℝ := 3104.35

/-- The amount of açaí berry juice added -/
def acai_amount : ℝ := 24.666666666666668

theorem mixed_fruit_juice_amount_correct :
  mixed_fruit_juice_amount * mixed_fruit_cost_per_litre +
  acai_amount * acai_cost_per_litre =
  (mixed_fruit_juice_amount + acai_amount) * superfruit_cost_per_litre :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_fruit_juice_amount_correct_l704_70450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_tangent_l704_70446

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the trajectory C
def on_trajectory (x y : ℝ) : Prop :=
  distance x y (-Real.sqrt 2) 0 + distance x y (Real.sqrt 2) 0 = 4

-- Define the circle
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the line y = 2
def on_line_y_2 (x y : ℝ) : Prop := y = 2

-- Define perpendicularity
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Define tangency
def is_tangent (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, (a * x + b * y + c = 0) → (x^2 + y^2 ≥ 2)

-- Theorem statement
theorem trajectory_and_tangent :
  (∀ x y : ℝ, on_trajectory x y ↔ x^2 / 4 + y^2 / 2 = 1) ∧
  (∀ x1 y1 x2 y2 : ℝ,
    on_trajectory x1 y1 →
    on_line_y_2 x2 y2 →
    perpendicular x1 y1 x2 y2 →
    ∃ a b c : ℝ, is_tangent a b c ∧ a * x1 + b * y1 + c = 0 ∧ a * x2 + b * y2 + c = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_tangent_l704_70446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l704_70425

open Set Real

noncomputable def f (x : ℝ) : ℝ := 1/((x - 1)*(x - 3)) + 1/((x - 3)*(x - 5)) + 1/((x - 5)*(x - 7))

theorem solution_set : {x : ℝ | f x = 1/12} = {13, -5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l704_70425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_implication_meaning_l704_70414

-- Define a custom proposition to represent "is a true proposition"
def is_true_proposition (p : Prop) : Prop := p

theorem implication_meaning (p q : Prop) :
  (p → q) ↔ (is_true_proposition (p → q)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_implication_meaning_l704_70414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spelling_contest_participants_l704_70440

theorem spelling_contest_participants 
  (eliminated_first : Real) 
  (continued_third : Real) 
  (third_round : Nat) 
  (original_participants : Nat) : Prop :=
  eliminated_first = 0.4 ∧
  continued_third = 1/4 ∧
  third_round = 30 ∧
  original_participants = 200 ∧
  (1 - eliminated_first) * continued_third * (original_participants : Real) = third_round

example : ∃ original_participants : Nat, 
  spelling_contest_participants 0.4 (1/4) 30 original_participants := by
  use 200
  simp [spelling_contest_participants]
  norm_num
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spelling_contest_participants_l704_70440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_conversion_l704_70419

theorem rectangular_to_polar_conversion :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r * (Real.cos θ) = Real.sqrt 2 ∧ r * (Real.sin θ) = -Real.sqrt 2 ∧
  r = 2 ∧ θ = 7 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_conversion_l704_70419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_spotlight_configuration_exists_l704_70470

/-- A spotlight that illuminates a convex shape in the arena -/
structure Spotlight where
  illuminated_area : Set (Real × Real)
  is_convex : Convex ℝ illuminated_area

/-- The circus arena -/
def Arena : Set (Real × Real) := Set.univ

/-- A configuration of spotlights in the arena -/
structure SpotlightConfiguration where
  n : Nat
  spotlights : Fin n → Spotlight
  covers_arena : ∀ x ∈ Arena, ∃ i, x ∈ (spotlights i).illuminated_area
  remains_covered_without_one : ∀ j, ∀ x ∈ Arena, ∃ i, i ≠ j ∧ x ∈ (spotlights i).illuminated_area
  not_covered_without_two : ∀ j k, j ≠ k → ∃ x ∈ Arena, ∀ i, i ≠ j → i ≠ k → x ∉ (spotlights i).illuminated_area

/-- The main theorem stating that a valid spotlight configuration exists for any n ≥ 2 -/
theorem valid_spotlight_configuration_exists (n : Nat) (h : n ≥ 2) :
  ∃ config : SpotlightConfiguration, config.n = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_spotlight_configuration_exists_l704_70470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_combination_count_l704_70463

/-- Represents the different types of coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- The value of each coin in cents --/
def coin_value : Coin → Nat
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- A combination of coins --/
def CoinCombination := List Coin

/-- Calculates the total value of a coin combination in cents --/
def combination_value (comb : CoinCombination) : Nat :=
  comb.map coin_value |>.sum

/-- Predicate to check if a combination sums to 50 cents --/
def is_valid_combination (comb : CoinCombination) : Prop :=
  combination_value comb = 50

/-- The set of all valid combinations --/
def valid_combinations : Set CoinCombination :=
  {comb | is_valid_combination comb}

/-- Theorem stating that there are 24 valid combinations --/
theorem coin_combination_count :
  ∃ (s : Finset CoinCombination), s.card = 24 ∧ ∀ comb, comb ∈ s ↔ is_valid_combination comb := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_combination_count_l704_70463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_determination_l704_70429

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem sine_function_determination (ω φ : ℝ) 
  (h1 : ω > 0)
  (h2 : -π/2 ≤ φ ∧ φ ≤ π/2)
  (h3 : ∃ (x₁ x₂ : ℝ), x₂ > x₁ ∧ 
    f ω φ x₂ - f ω φ x₁ = 2 * Real.sqrt 2 ∧ 
    (∀ (x : ℝ), x₁ < x ∧ x < x₂ → f ω φ x ≤ f ω φ x₂))
  (h4 : f ω φ 2 = -1/2) :
  ω = π/2 ∧ φ = π/6 := by
  sorry

#check sine_function_determination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_determination_l704_70429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l704_70409

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * (x + Real.pi / 12))

-- Define the axis of symmetry
noncomputable def axis_of_symmetry : ℝ := Real.pi / 6

-- Theorem statement
theorem symmetry_axis_of_f :
  ∀ x : ℝ, f (axis_of_symmetry - x) = f (axis_of_symmetry + x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l704_70409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_COD_area_l704_70477

/-- Given a triangle COD where C(0,p) lies on the y-axis and D(x,0) lies on the x-axis,
    with x and p being positive integers and D located between O(0,0) and B(12,0),
    prove that the area of triangle COD is xp/2. -/
theorem triangle_COD_area (x p : ℕ) (h1 : 0 < x) (h2 : x ≤ 12) (h3 : 0 < p) :
  (1/2 : ℝ) * x * p = x * p / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_COD_area_l704_70477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l704_70448

-- Define the function f(x) = x^2 - 2x
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the domain [0, 4]
def domain : Set ℝ := Set.Icc 0 4

-- Theorem stating that f is monotonically decreasing on (0, 1)
theorem f_monotone_decreasing :
  ∀ x y, x ∈ Set.Ioo 0 1 ∩ domain → y ∈ Set.Ioo 0 1 ∩ domain → x < y → f x > f y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l704_70448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_and_leading_coeff_l704_70456

-- Define the complex number 3+2i
def z : ℂ := 3 + 2*Complex.I

-- Define the quadratic polynomial
def f (x : ℂ) : ℂ := 2*x^2 - 12*x + 26

theorem quadratic_root_and_leading_coeff :
  f z = 0 ∧ (∀ x : ℂ, ∃ y : ℂ, f x = 2*x^2 + y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_and_leading_coeff_l704_70456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_zero_instants_l704_70427

-- Define the motion equation
noncomputable def s (t : ℝ) : ℝ := (1/4) * t^4 - 4 * t^3 + 16 * t^2

-- Define the velocity function (derivative of s)
noncomputable def v (t : ℝ) : ℝ := t^3 - 12 * t^2 + 32 * t

-- Theorem stating the instants when velocity is zero
theorem velocity_zero_instants : 
  {t : ℝ | v t = 0} = {0, 4, 8} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_zero_instants_l704_70427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_properties_l704_70458

def a : ℕ → ℚ
  | 0 => 1/2
  | n+1 => 1/2 + (a n)^2 / 2

theorem a_properties : ∀ n : ℕ,
  (a n < a (n + 1)) ∧ (a n < 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_properties_l704_70458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_approximation_l704_70498

/-- The closest approximation of (69.28 × 0.004) / 0.03 to two decimal places is 9.24 -/
theorem closest_approximation : 
  let x := (69.28 * 0.004) / 0.03
  ∀ y : ℝ, y ≠ 9.24 → (∃ (n : ℤ), y = n / 100) → 
    abs (x - 9.24) ≤ abs (x - y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_approximation_l704_70498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_4_equals_6_l704_70406

-- Define the function f as noncomputable
noncomputable def f (y : ℝ) : ℝ := 
  let x := (y - 1) / 3
  x^2 + 3*x + 2

-- Theorem statement
theorem f_of_4_equals_6 : f 4 = 6 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_4_equals_6_l704_70406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l704_70432

theorem unique_solution_exponential_equation :
  ∃! (x y : ℝ), (9 : ℝ)^(x^2 + y) + (9 : ℝ)^(x + y^2) = 1 ∧ x = -1/2 ∧ y = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l704_70432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_implies_a_eq_one_monotone_decreasing_implies_a_leq_six_fifths_l704_70433

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x * f a x

-- Theorem for part 1
theorem extreme_point_implies_a_eq_one (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ∈ Set.Ioo (2 - ε) (2 + ε) → f a x ≤ f a 2) →
  a = 1 := by sorry

-- Theorem for part 2
theorem monotone_decreasing_implies_a_leq_six_fifths (a : ℝ) :
  (∀ (x y : ℝ), x ∈ Set.Icc 0 2 → y ∈ Set.Icc 0 2 → x ≤ y → g a y ≤ g a x) →
  a ≤ 6/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_implies_a_eq_one_monotone_decreasing_implies_a_leq_six_fifths_l704_70433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_effect_on_income_l704_70422

/-- Represents a store's sales data before and after a discount -/
structure StoreSales where
  originalPrice : ℝ
  originalQuantity : ℝ
  discountPercent : ℝ
  quantityIncreasePercent : ℝ

/-- Calculates the percentage increase in gross income after applying a discount -/
noncomputable def grossIncomeIncreasePercent (s : StoreSales) : ℝ :=
  let newPrice := s.originalPrice * (1 - s.discountPercent)
  let newQuantity := s.originalQuantity * (1 + s.quantityIncreasePercent)
  let originalIncome := s.originalPrice * s.originalQuantity
  let newIncome := newPrice * newQuantity
  (newIncome - originalIncome) / originalIncome * 100

/-- Theorem stating that a 10% discount with a 20% increase in quantity results in an 8% increase in gross income -/
theorem discount_effect_on_income (s : StoreSales) 
    (h1 : s.discountPercent = 0.1) 
    (h2 : s.quantityIncreasePercent = 0.2) : 
    grossIncomeIncreasePercent s = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_effect_on_income_l704_70422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l704_70401

-- Define the ellipse
noncomputable def is_on_ellipse (x y : ℝ) : Prop :=
  y^2 / 16 + x^2 / 9 = 1

-- Define the line
noncomputable def line (x m : ℝ) : ℝ := x + m

-- Define the distance function between a point and a line
noncomputable def distance_point_to_line (x y m : ℝ) : ℝ :=
  |y - line x m| / Real.sqrt 2

-- State the theorem
theorem min_m_value (m : ℝ) :
  (∃ x y : ℝ, is_on_ellipse x y ∧ distance_point_to_line x y m = Real.sqrt 2) →
  (∀ m' : ℝ, (∃ x' y' : ℝ, is_on_ellipse x' y' ∧ distance_point_to_line x' y' m' = Real.sqrt 2) → m' ≥ m) →
  m = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l704_70401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_quantity_l704_70465

/-- The annual purchase amount in tons -/
noncomputable def annual_purchase : ℝ := 600

/-- The shipping cost per purchase in thousand yuan -/
noncomputable def shipping_cost_per_purchase : ℝ := 60

/-- The storage cost coefficient in thousand yuan per ton -/
noncomputable def storage_cost_coefficient : ℝ := 4

/-- The total cost function in thousand yuan -/
noncomputable def total_cost (x : ℝ) : ℝ := 
  (annual_purchase / x) * shipping_cost_per_purchase + storage_cost_coefficient * x

/-- Theorem stating that the optimal purchase quantity is 30 tons -/
theorem optimal_purchase_quantity :
  ∃ (x : ℝ), x > 0 ∧ x = 30 ∧ ∀ (y : ℝ), y > 0 → total_cost x ≤ total_cost y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_quantity_l704_70465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l704_70403

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos (x + Real.pi / 2))^2 + Real.sin (2 * x + Real.pi / 6) - 1

theorem f_properties :
  (∀ x, f x ≤ 1) ∧
  (∀ x, f x = Real.sin (2 * x - Real.pi / 6)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l704_70403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_lower_bound_l704_70490

def f (n : ℕ) : ℚ := (Finset.range n).sum (λ k ↦ 1 / (k + 1 : ℚ))

theorem harmonic_sum_lower_bound (n : ℕ) : 
  f (2^(n + 1)) > (n + 3 : ℚ) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_lower_bound_l704_70490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l704_70469

def sequence_stabilizes (a b c : ℕ → ℕ) : Prop :=
  ∃ k : ℕ, a (k + 1) = a k ∧ b (k + 1) = b k ∧ c (k + 1) = c k

theorem sequence_convergence
  (a b c : ℕ → ℕ)
  (h1 : ∀ n, a (n + 1) = Int.natAbs (b n - c n))
  (h2 : ∀ n, b (n + 1) = Int.natAbs (c n - a n))
  (h3 : ∀ n, c (n + 1) = Int.natAbs (a n - b n))
  (h4 : a 1 > 0 ∧ b 1 > 0 ∧ c 1 > 0) :
  sequence_stabilizes a b c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l704_70469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janice_weekly_earnings_l704_70493

/-- Calculates Janice's weekly earnings based on her work hours and pay rates --/
def janice_earnings (weekday_hours : ℕ) (weekend_hours : ℕ) (holiday_hours : ℕ) 
  (weekday_rate : ℕ) (weekend_rate : ℕ) (weekday_overtime_rate : ℕ) 
  (weekend_overtime_rate : ℕ) (cases_completed : ℕ) : ℤ :=
  let regular_hours := 40
  let weekday_earnings := (min weekday_hours regular_hours * weekday_rate : ℤ)
  let weekend_regular_hours := min weekend_hours (regular_hours - weekday_hours)
  let weekend_regular_earnings := (weekend_regular_hours * weekend_rate : ℤ)
  let weekend_overtime_hours := max (weekend_hours - weekend_regular_hours) 0
  let weekend_overtime_earnings := (weekend_overtime_hours * weekend_overtime_rate : ℤ)
  let holiday_rate := 2 * weekend_overtime_rate
  let holiday_earnings := (holiday_hours * holiday_rate : ℤ)
  let total_earnings := weekday_earnings + weekend_regular_earnings + 
                        weekend_overtime_earnings + holiday_earnings
  let bonus := if cases_completed ≥ 20 then 50 else if cases_completed ≤ 15 then -30 else 0
  total_earnings + bonus

/-- Theorem stating that Janice's earnings for the given week are $870 --/
theorem janice_weekly_earnings : 
  janice_earnings 30 25 5 10 12 15 18 17 = 870 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_janice_weekly_earnings_l704_70493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_equals_142_6_l704_70408

/-- Rounds a real number to the nearest tenth -/
noncomputable def round_to_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The sum of 95.32 and 47.268, rounded to the nearest tenth, equals 142.6 -/
theorem sum_and_round_equals_142_6 :
  round_to_tenth (95.32 + 47.268) = 142.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_equals_142_6_l704_70408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_platform_length_theorem_l704_70462

/-- Given a train crossing two platforms, this theorem proves the length of the first platform. -/
theorem first_platform_length_theorem
  (train_length : ℝ)
  (first_platform_time : ℝ)
  (second_platform_time : ℝ)
  (second_platform_length : ℝ)
  (h1 : train_length = 30)
  (h2 : first_platform_time = 15)
  (h3 : second_platform_time = 20)
  (h4 : second_platform_length = 250)
  (h5 : first_platform_time / second_platform_time = (train_length + first_platform_length) / (train_length + second_platform_length)) :
  first_platform_length = 180 :=
by sorry

/-- The length of the first platform that the train crosses. -/
def first_platform_length : ℝ := 180

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_platform_length_theorem_l704_70462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l704_70468

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h1 : S seq 6 > S seq 7) (h2 : S seq 7 > S seq 5) :
  seq.d < 0 ∧ 
  S seq 11 > 0 ∧ 
  (∀ n > 12, S seq n ≤ 0) ∧
  (∀ n ≤ 12, S seq n > 0) ∧
  |seq.a 6| > |seq.a 7| := by
  sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l704_70468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_curve_l704_70457

-- Define the curve
def on_curve (Q : ℝ × ℝ) : Prop :=
  (Q.1^2 + (Q.2 + 2)^2 = 1)

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The theorem
theorem min_distance_to_curve :
  ∃ m : ℝ, ∀ P Q : ℝ × ℝ, on_curve Q → distance P Q ≥ m ∧
  ∀ m' : ℝ, (∀ P Q : ℝ × ℝ, on_curve Q → distance P Q ≥ m') → m ≥ m' :=
by
  sorry

#check min_distance_to_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_curve_l704_70457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_plus_cube_root_sum_l704_70455

theorem square_root_plus_cube_root_sum : ∃ x y : ℝ,
  x = Real.sqrt ((- Real.sqrt 9) ^ 2) ∧
  y = (64 : ℝ) ^ (1/3) ∧
  x + y = 7 := by
  -- Introduce x and y
  let x := Real.sqrt ((- Real.sqrt 9) ^ 2)
  let y := (64 : ℝ) ^ (1/3)
  
  -- Prove existence
  use x, y
  
  -- Prove the conditions
  constructor
  · rfl
  constructor
  · rfl
  · -- The actual proof would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_plus_cube_root_sum_l704_70455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kelly_buys_ten_pounds_l704_70494

/-- The number of pounds of mangoes Kelly can buy -/
noncomputable def mangoes_kelly_can_buy (half_pound_cost : ℝ) (kelly_budget : ℝ) : ℝ :=
  kelly_budget / (2 * half_pound_cost)

/-- Theorem: Kelly can buy 10 pounds of mangoes -/
theorem kelly_buys_ten_pounds (half_pound_cost : ℝ) (kelly_budget : ℝ) 
  (h1 : half_pound_cost = 0.60)
  (h2 : kelly_budget = 12) : 
  mangoes_kelly_can_buy half_pound_cost kelly_budget = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kelly_buys_ten_pounds_l704_70494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_proofs_l704_70438

def num_balls : ℕ := 5
def num_boxes : ℕ := 4

theorem ball_distribution_proofs :
  (num_balls^num_boxes = 4^5) ∧
  (Nat.choose num_balls 4 * Nat.choose num_boxes 1 = Nat.choose 5 4 * Nat.choose 4 1) ∧
  (Nat.choose num_balls 2 * Nat.factorial num_boxes = Nat.choose 5 2 * Nat.factorial 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_proofs_l704_70438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_l704_70402

def mySequence (a : ℕ → ℚ) : Prop :=
  (a 1 = 3) ∧ 
  (a 3 = 26) ∧ 
  ∀ n > 1, a n = (1 : ℚ) / 3 * (a (n-1) + a (n+1))

theorem sixth_term (a : ℕ → ℚ) (h : mySequence a) : a 6 = 278 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_l704_70402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l704_70460

/-- Given a sequence {a_n} with the sum of the first n terms S_n = an^2 + bn,
    where a < 0, prove that na_n ≤ S_n ≤ na_1 for all n ∈ ℕ. -/
theorem sequence_inequality (a b : ℝ) (a_n : ℕ → ℝ) (h : a < 0) :
  ∀ n : ℕ, n * a_n n ≤ (a * n^2 + b * n) ∧ (a * n^2 + b * n) ≤ n * a_n 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l704_70460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l704_70473

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the distance from a point to the y-axis
def dist_to_y_axis (x : ℝ) : ℝ := |x|

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem parabola_focus_distance (x y : ℝ) : 
  parabola x y → dist_to_y_axis x = 4 → distance x y (focus.1) (focus.2) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l704_70473
