import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_log_decreasing_l497_49779

-- Define the function f(x) = log_a(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem contrapositive_log_decreasing (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y, x < y → f a x > f a y) →
  (f a 2 < 0) ↔
  (f a 2 ≥ 0) →
  ¬(∀ x y, x < y → f a x > f a y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_log_decreasing_l497_49779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_y_150_divided_by_y_minus_2_to_4_l497_49775

theorem remainder_y_150_divided_by_y_minus_2_to_4 (y : ℤ) :
  ∃ q : Polynomial ℤ, y^150 = (X - 2)^4 * q + (554350 * (X-2)^3 + 22350 * (X-2)^2 + 600 * (X-2) + 8 * 2^147) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_y_150_divided_by_y_minus_2_to_4_l497_49775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_is_ten_l497_49711

-- Define the side length of the square
noncomputable def square_side : ℝ := Real.sqrt 2025

-- Define the radius of the circle
noncomputable def circle_radius : ℝ := square_side

-- Define the breadth of the rectangle
noncomputable def rectangle_breadth : ℝ := (3 / 5) * circle_radius

-- Define the area of the rectangle
def rectangle_area : ℝ := 270

-- State the theorem
theorem rectangle_length_is_ten :
  ∃ (length : ℝ), length * rectangle_breadth = rectangle_area ∧ length = 10 := by
  -- Proof goes here
  sorry

#eval rectangle_area -- This will work
-- #eval rectangle_breadth -- This won't work due to being noncomputable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_is_ten_l497_49711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_projection_theorem_l497_49777

/-- Dot product of two 2D vectors -/
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

/-- Given two vectors in R², prove that their common projection onto some vector is (3, 3) -/
theorem common_projection_theorem : 
  ∃ (v p : ℝ × ℝ), 
    (∀ (c : ℝ), c • v = p ∨ c • v ≠ p) ∧ 
    (∃ (c₁ c₂ : ℝ), c₁ • v = p ∧ c₂ • v = p) ∧
    (dot_product (4, 2) v / dot_product v v) • v = p ∧
    (dot_product (1, 5) v / dot_product v v) • v = p ∧
    p = (3, 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_projection_theorem_l497_49777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_characterization_l497_49715

/-- The set of polynomials of degree n with coefficients being a permutation of {2^0, 2^1, ..., 2^n} -/
def P (n : ℕ) : Set (Polynomial ℤ) :=
  {p : Polynomial ℤ | p.degree = n ∧ 
    ∃ σ : Fin (n+1) ≃ Fin (n+1), ∀ i, p.coeff i = (2 : ℤ) ^ (σ i : ℕ)}

/-- The main theorem -/
theorem polynomial_divisibility_characterization :
  ∀ k d : ℕ, (∃ n : ℕ, ∀ p ∈ P n, (p.eval (k : ℤ)) % (d : ℤ) = 0) ↔
  (∃ a b : ℕ, k = b * (2*a+1) + 1 ∧ d = 2*a+1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_characterization_l497_49715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_seven_l497_49720

/-- An arithmetic sequence -/
structure ArithmeticSequence (α : Type*) [Add α] [Mul α] where
  a : ℕ → α
  d : α
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence ℝ) (n : ℕ) : ℝ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_sum_seven
  (seq : ArithmeticSequence ℝ)
  (h : seq.a 3 + seq.a 5 = 14) :
  sum_n seq 7 = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_seven_l497_49720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_vector_coordinates_l497_49750

/-- Given points A(-2,3,5) and B(1,-1,-7), prove that the opposite vector of AB has coordinates (-3,4,12). -/
theorem opposite_vector_coordinates :
  let A : Fin 3 → ℝ := ![(-2), 3, 5]
  let B : Fin 3 → ℝ := ![1, (-1), (-7)]
  let AB : Fin 3 → ℝ := ![B 0 - A 0, B 1 - A 1, B 2 - A 2]
  let opposite_AB : Fin 3 → ℝ := ![-AB 0, -AB 1, -AB 2]
  opposite_AB = ![(-3), 4, 12] := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_vector_coordinates_l497_49750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_multiplication_l497_49776

theorem square_area_multiplication (side_length : Real) (h : side_length = Real.exp (Real.log 10 * 0.2)) :
  let area := side_length^2
  area * Real.exp (Real.log 10 * 0.1) * Real.exp (Real.log 10 * (-0.3)) * Real.exp (Real.log 10 * 0.4) = Real.exp (Real.log 10 * 0.6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_multiplication_l497_49776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_tiling_l497_49763

/-- A chessboard is an 8x8 grid of squares. -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- A 4x1 rectangle covers 4 consecutive squares in a row or column. -/
def Rectangle4x1 := Fin 4 → Fin 1 → Bool

/-- A 2x2 square covers a 2x2 area on the chessboard. -/
def Square2x2 := Fin 2 → Fin 2 → Bool

/-- Check if a 4x1 rectangle covers a specific square on the chessboard. -/
def covers4x1 (rect : Rectangle4x1) (i j : Fin 8) : Prop :=
  ∃ (k : Fin 4) (l : Fin 1), i = k ∧ j = l

/-- Check if a 2x2 square covers a specific square on the chessboard. -/
def covers2x2 (square : Square2x2) (i j : Fin 8) : Prop :=
  ∃ (k l : Fin 2), i = k ∧ j = l

/-- A tiling is a placement of rectangles and squares on the chessboard. -/
def Tiling (board : Chessboard) (rect : Fin 15 → Rectangle4x1) (square : Square2x2) : Prop :=
  ∀ i j, ∃ k, (k < 15 ∧ covers4x1 (rect k) i j) ∨ covers2x2 square i j

/-- The main theorem: it's impossible to tile the chessboard with 15 4x1 rectangles and one 2x2 square. -/
theorem impossible_tiling :
  ¬∃ (board : Chessboard) (rect : Fin 15 → Rectangle4x1) (square : Square2x2),
    Tiling board rect square := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_tiling_l497_49763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_ends_after_27_rounds_l497_49784

/-- Represents the state of the game at any given round -/
structure GameState where
  a : ℕ  -- tokens of player A
  b : ℕ  -- tokens of player B
  c : ℕ  -- tokens of player C
deriving Inhabited

/-- Represents the rules of the game -/
def next_state (s : GameState) : GameState :=
  if s.a ≥ s.b ∧ s.a ≥ s.c then
    { a := s.a - 4, b := s.b + 1, c := s.c + 1 }
  else if s.b ≥ s.a ∧ s.b ≥ s.c then
    { a := s.a + 1, b := s.b - 4, c := s.c + 1 }
  else
    { a := s.a + 1, b := s.b + 1, c := s.c - 4 }

/-- Checks if the game has ended -/
def game_ended (s : GameState) : Prop :=
  s.a = 0 ∨ s.b = 0 ∨ s.c = 0

/-- The initial state of the game -/
def initial_state : GameState :=
  { a := 10, b := 9, c := 8 }

/-- The theorem stating that the game ends after 27 rounds -/
theorem game_ends_after_27_rounds :
  ∃ (states : List GameState),
    states.length = 28 ∧
    states.head! = initial_state ∧
    (∀ i, i < 27 → states[i + 1]! = next_state states[i]!) ∧
    game_ended states[27]! := by
  sorry

#check game_ends_after_27_rounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_ends_after_27_rounds_l497_49784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_time_l497_49770

/-- Represents the time taken for a boat to travel downstream given upstream travel time and speed ratio -/
noncomputable def downstream_time (upstream_time : ℝ) (speed_ratio : ℝ) : ℝ :=
  (upstream_time * (speed_ratio - 1)) / (speed_ratio + 1)

/-- Theorem stating that for a boat traveling upstream for 6 hours with a speed ratio of 4:1 to the current,
    the time taken to cover the same distance downstream is 3.6 hours -/
theorem boat_downstream_time :
  downstream_time 6 4 = 3.6 := by
  -- Unfold the definition of downstream_time
  unfold downstream_time
  -- Simplify the expression
  simp
  -- The proof is completed with 'sorry' as requested
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_time_l497_49770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_intersection_l497_49705

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + t, -1/2 * t)

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

-- Define the general equation of line l
def line_l_eq (x y : ℝ) : Prop := x + 2 * y - 1 = 0

-- Define the general equation of curve C
def curve_C_eq (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Theorem statement
theorem midpoint_of_intersection :
  ∃ (t₁ t₂ : ℝ),
    let (x₁, y₁) := line_l t₁
    let (x₂, y₂) := line_l t₂
    curve_C_eq x₁ y₁ ∧ curve_C_eq x₂ y₂ ∧
    (x₁ + x₂) / 2 = 1/2 ∧ (y₁ + y₂) / 2 = 1/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_intersection_l497_49705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_ends_in_14_rounds_l497_49737

/-- Represents the state of a tournament after a round -/
structure TournamentState where
  noLoss : Nat
  oneLoss : Nat

/-- Simulates a round of the tournament -/
def simulateRound (state : TournamentState) : TournamentState :=
  { noLoss := state.noLoss / 2,
    oneLoss := state.noLoss / 2 + state.oneLoss / 2 }

/-- Checks if the tournament can continue -/
def canContinue (state : TournamentState) : Bool :=
  state.noLoss + state.oneLoss ≥ 2

/-- Simulates the entire tournament -/
def simulateTournament (initialState : TournamentState) : Nat :=
  let rec simulate (state : TournamentState) (rounds : Nat) (fuel : Nat) : Nat :=
    match fuel with
    | 0 => rounds
    | fuel + 1 =>
      if canContinue state then
        simulate (simulateRound state) (rounds + 1) fuel
      else
        rounds
  simulate initialState 0 (initialState.noLoss + initialState.oneLoss)

/-- The main theorem to prove -/
theorem tournament_ends_in_14_rounds :
  simulateTournament { noLoss := 1152, oneLoss := 0 } = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_ends_in_14_rounds_l497_49737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bilingual_students_l497_49719

/-- Given a survey of students about their language skills, we prove the number of bilingual students. -/
theorem bilingual_students (total : ℕ) (non_french_percent : ℚ) (french_non_english : ℕ) 
  (h1 : total = 500)
  (h2 : non_french_percent = 86 / 100)
  (h3 : french_non_english = 40) : 
  total - (total * non_french_percent).floor - french_non_english = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bilingual_students_l497_49719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_denominator_infinite_geometric_sum_denominator_l497_49733

/-- The sum of a geometric progression with n terms, first term a, and common ratio r -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of an infinite convergent geometric progression with first term a and common ratio r -/
noncomputable def infinite_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

/-- The function of the common ratio that appears in the denominator of both finite and infinite geometric sums -/
def common_ratio_function (r : ℝ) : ℝ :=
  1 - r

theorem geometric_sum_denominator (a : ℝ) (r : ℝ) (n : ℕ) (h : r ≠ 1) :
  ∃ k, geometric_sum a r n = k / common_ratio_function r := by
  sorry

theorem infinite_geometric_sum_denominator (a : ℝ) (r : ℝ) (h : |r| < 1) :
  ∃ k, infinite_geometric_sum a r = k / common_ratio_function r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_denominator_infinite_geometric_sum_denominator_l497_49733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_height_intersection_l497_49734

/-- Represents the height of a ball as a function of time -/
def ballHeight (a b c : ℝ) (t : ℝ) : ℝ := a * t^2 + b * t + c

theorem ball_height_intersection (a b c : ℝ) :
  (∀ t, ballHeight a b c 1.1 ≥ ballHeight a b c t) →  -- maximum height at 1.1 seconds
  (ballHeight a b c 0 = ballHeight a b c (-1)) →        -- same initial height
  (∃ t, ballHeight a b c t = ballHeight a b c (t - 1)) → -- heights are equal at some point
  (∃ t, t = 1.6 ∧ ballHeight a b c t = ballHeight a b c (t - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_height_intersection_l497_49734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l497_49791

def M : Set ℝ := {x | ∃ k : ℤ, x = 45 + k * 90}
def N : Set ℝ := {y | ∃ k : ℤ, y = 90 + k * 45}

theorem angle_properties :
  (∀ m : ℝ, m > 0 → {α : ℝ | ∃ k : ℤ, α = π/4 + 2*k*π} = {α : ℝ | ∃ (x y : ℝ), x = m ∧ y = m ∧ α = Real.arctan (y/x)}) ∧
  ({x : ℝ | x ∈ Set.Icc 0 (2*π) ∧ Real.tan x + Real.sin x < 0} = Set.Ioo (π/2) π ∪ Set.Ioo (3*π/2) (2*π)) ∧
  (M ⊆ N) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l497_49791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_equality_l497_49794

/-- Given a function g and its inverse, prove the value of b -/
theorem inverse_function_equality (b : ℝ) (g : ℝ → ℝ) : 
  (∀ x, g x = 1 / (3 * x + b)) → 
  (∀ x, Function.invFun g x = (1 - 3 * x) / (3 * x)) → 
  b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_equality_l497_49794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l497_49761

noncomputable def f (x : ℝ) : ℝ := (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7))

def solution_set : Set ℝ :=
  Set.Ioo 2 3 ∪ Set.Ioo 4 5 ∪ Set.Ioo 6 7 ∪ Set.Ioi 7

theorem inequality_solution (x : ℝ) : f x > 0 ↔ x ∈ solution_set := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l497_49761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marie_biking_speed_l497_49766

/-- Given Marie's biking distance and time, prove her speed is approximately 12.0 miles per hour -/
theorem marie_biking_speed (distance : ℝ) (time : ℝ) (h1 : distance = 31.0) (h2 : time = 2.583333333) :
  abs ((distance / time) - 12.0) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marie_biking_speed_l497_49766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_statements_l497_49751

-- Define the properties of irrational numbers
noncomputable def is_irrational (x : ℝ) : Prop := ¬ (∃ (q : ℚ), x = ↑q)

-- Define the statements about irrational numbers
def statement1 : Prop := ∀ x : ℝ, is_irrational x → ¬ (∃ y : ℝ, x = Real.sqrt y)
def statement2 : Prop := ∀ x : ℝ, is_irrational x → ∀ n : ℕ, ∃ m > n, (Int.floor (x * 10^m) : ℤ) ≠ (Int.floor (x * 10^n) : ℤ) * 10^(m-n)
def statement3 : Prop := is_irrational 0 ∧ (∀ x : ℝ, is_irrational x → is_irrational (-x))
def statement4 : Prop := ∀ x : ℝ, is_irrational x → ∃ y : ℝ, x = y

theorem irrational_statements :
  (statement2 ∧ statement4) ∧ (¬statement1 ∧ ¬statement3) := by
  sorry

#check irrational_statements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_statements_l497_49751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_midpoint_sums_theorem_l497_49790

/-- Represents a labeling of a cube's vertices -/
def CubeLabeling := Fin 8 → ℕ

/-- The set of all possible sums at the midpoints of a cube's edges -/
def midpointSums (labeling : CubeLabeling) : Finset ℕ :=
  Finset.image (λ (p : Fin 8 × Fin 8) => labeling p.1 + labeling p.2)
    (Finset.filter (λ (p : Fin 8 × Fin 8) => p.1 < p.2) (Finset.univ.product Finset.univ))

/-- A valid labeling uses each number from 1 to 8 exactly once -/
def isValidLabeling (labeling : CubeLabeling) : Prop :=
  (∀ n : ℕ, n ∈ Finset.range 8 → ∃ i : Fin 8, labeling i = n + 1) ∧
  (∀ i j : Fin 8, i ≠ j → labeling i ≠ labeling j)

theorem cube_midpoint_sums_theorem :
  (∃ (labeling : CubeLabeling), isValidLabeling labeling ∧ (midpointSums labeling).card = 5) ∧
  (¬ ∃ (labeling : CubeLabeling), isValidLabeling labeling ∧ (midpointSums labeling).card = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_midpoint_sums_theorem_l497_49790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_of_seven_equals_fiftyone_l497_49788

-- Define the derivative operation
def my_deriv (x : ℝ) : ℝ := 3 * x - 3

-- Theorem statement
theorem second_derivative_of_seven_equals_fiftyone :
  my_deriv (my_deriv 7) = 51 := by
  -- Compute the first derivative
  have h1 : my_deriv 7 = 18 := by
    unfold my_deriv
    norm_num
  
  -- Compute the second derivative
  have h2 : my_deriv 18 = 51 := by
    unfold my_deriv
    norm_num
  
  -- Combine the results
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_of_seven_equals_fiftyone_l497_49788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_interval_of_abs_tan_l497_49768

open Real

theorem monotone_decreasing_interval_of_abs_tan (k : ℤ) :
  let f : ℝ → ℝ := λ x ↦ |tan (x / 2 - π / 6)|
  ∀ x₁ x₂, x₁ ∈ Set.Ioo (2 * k * π - 2 * π / 3) (2 * k * π + π / 3) →
           x₂ ∈ Set.Ioo (2 * k * π - 2 * π / 3) (2 * k * π + π / 3) →
           x₁ < x₂ → f x₁ ≥ f x₂ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_interval_of_abs_tan_l497_49768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_in_simplest_form_l497_49708

/-- A fraction is in its simplest form if its numerator and denominator have no common factors other than 1 -/
def IsSimplestForm {R : Type*} [CommRing R] (num den : R) : Prop :=
  ∀ (f : R), f ∣ num ∧ f ∣ den → IsUnit f

theorem fraction_in_simplest_form {R : Type*} [CommRing R] (x y : R) :
  IsSimplestForm (x^2 + y^2) (x + y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_in_simplest_form_l497_49708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l497_49760

noncomputable def g (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 8 * Real.sin x ^ 2 + 2 * Real.sin x + 2 * Real.cos x ^ 2 - 10) / (Real.sin x - 1)

theorem g_range :
  Set.range (fun (x : ℝ) => g x) = Set.Ico 3 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l497_49760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_count_l497_49740

/-- Represents the number of sides in a convex polygon --/
def n : ℕ := 12

/-- Represents the common difference in the arithmetic progression of interior angles --/
def common_difference : ℝ := 10

/-- Represents the largest interior angle of the polygon --/
def largest_angle : ℝ := 175

/-- The sum of interior angles of a polygon with n sides --/
noncomputable def sum_of_interior_angles : ℝ := 180 * (n - 2)

/-- The sum of angles in an arithmetic sequence --/
noncomputable def sum_of_arithmetic_sequence : ℝ := n * (largest_angle - common_difference * (n - 1) / 2 + largest_angle) / 2

theorem polygon_sides_count : sum_of_interior_angles = sum_of_arithmetic_sequence → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_count_l497_49740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_problem_l497_49707

/-- Given two solutions x and y, where x is 10% alcohol by volume,
    and 150 mL of y is added to 50 mL of x to create a solution that is 25% alcohol,
    prove that y is 30% alcohol by volume. -/
theorem alcohol_mixture_problem (y : ℝ) : 
  (10 / 100 * 50 + y / 100 * 150) / 200 * 100 = 25 →
  y = 30 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_problem_l497_49707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l497_49721

-- Define the triangle and points
structure Triangle (A B D : Point) where
  mk :: -- Empty structure, just to define the concept

-- Define the angle measure function
noncomputable def angle_measure (p q r : Point) : ℝ := sorry

-- Define the "lies on" relation
def lies_on (C : Point) (B D : Point) : Prop := sorry

-- State the theorem
theorem triangle_angle_measure 
  (A B C D : Point) 
  (tri : Triangle A B D) 
  (h1 : angle_measure A B D = 28)
  (h2 : angle_measure D B C = 46)
  (h3 : lies_on C B D)
  (h4 : angle_measure B A C = 30) :
  angle_measure C B D = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l497_49721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_valid_sequence_l497_49718

/-- Represents a tile color -/
inductive Color
| Black
| White

/-- Represents a configuration of tiles on an n × n grid -/
def Configuration (n : ℕ) := Fin n → Fin n → Color

/-- Helper function to get adjacent white tiles -/
def AdjacentWhiteTiles (n : ℕ) (config : Configuration n) 
  (i j : Fin n) : Finset (Fin n × Fin n) :=
sorry

/-- Helper function to get adjacent black tiles -/
def AdjacentBlackTiles (n : ℕ) (config : Configuration n) 
  (i j : Fin n) : Finset (Fin n × Fin n) :=
sorry

/-- Checks if a configuration satisfies the placement rules -/
def ValidConfiguration (n : ℕ) (config : Configuration n) : Prop :=
  ∀ i j : Fin n,
    (config i j = Color.Black → 
      (AdjacentWhiteTiles n config i j).card % 2 = 0) ∧
    (config i j = Color.White → 
      (AdjacentBlackTiles n config i j).card % 2 = 1)

/-- Represents the process of placing layers -/
def LayerPlacementProcess (n : ℕ) (initial : Configuration n) : 
  ℕ → Configuration n :=
sorry

/-- Theorem: There is no infinite sequence of valid configurations -/
theorem no_infinite_valid_sequence (n : ℕ) :
  ¬∃ (initial : Configuration n),
    ∀ k : ℕ, ValidConfiguration n (LayerPlacementProcess n initial k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_valid_sequence_l497_49718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_independence_l497_49727

theorem function_independence (f : ℤ → ℤ) : 
  (∀ g : ℤ → ℤ, ∃ k : ℤ, ∀ n : ℤ, f (g n) - g (f n) = k) →
  (∀ x : ℤ, f x = x) ∨ (∃ c : ℤ, ∀ x : ℤ, f x = c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_independence_l497_49727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_valid_triples_l497_49762

def a : ℕ := 1944

-- Define the property of being mutually coprime
def mutually_coprime (x y z : ℕ) : Prop :=
  Nat.Coprime x y ∧ Nat.Coprime x z ∧ Nat.Coprime y z

-- Define the property of being a valid triple
def valid_triple (x y : ℕ) : Prop :=
  x ∣ a ∧ y ∣ a ∧ (x + y) ∣ a ∧
  mutually_coprime x y (x + y) ∧
  (∀ p : ℕ, Nat.Prime p → p > 1 → (p ∣ x ∧ p ∣ y) → p ∣ (x + y)) ∧
  (∀ p : ℕ, Nat.Prime p → p > 1 → (p ∣ x ∧ p ∣ (x + y)) → p ∣ y) ∧
  (∀ p : ℕ, Nat.Prime p → p > 1 → (p ∣ y ∧ p ∣ (x + y)) → p ∣ x)

-- The main theorem
theorem max_product_of_valid_triples :
  (∃ x y : ℕ, valid_triple x y ∧
    ∀ x' y' : ℕ, valid_triple x' y' →
      x * y * (x + y) ≥ x' * y' * (x' + y')) ∧
  (∃ x y : ℕ, valid_triple x y ∧ x * y * (x + y) = 2^7 * 3^15) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_valid_triples_l497_49762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_logarithmic_expressions_l497_49738

theorem order_of_logarithmic_expressions :
  let a := Real.log 2
  let b := Real.exp 2.1
  let c := Real.log (2 / 3)
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_logarithmic_expressions_l497_49738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l497_49704

/-- The line equation y = (x + 3) / 2 -/
noncomputable def line_equation (x : ℝ) : ℝ := (x + 3) / 2

/-- The point we're measuring distance to -/
def target_point : ℝ × ℝ := (8, 3)

/-- The point on the line claimed to be closest to the target point -/
def closest_point : ℝ × ℝ := (7, 5)

/-- Theorem stating that the closest_point is indeed the closest point on the line to the target_point -/
theorem closest_point_on_line : 
  ∀ (x : ℝ), 
  (closest_point.1 - target_point.1)^2 + (closest_point.2 - target_point.2)^2 
  ≤ (x - target_point.1)^2 + (line_equation x - target_point.2)^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l497_49704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_decrease_percent_l497_49729

-- Define the original tax and consumption
variable (T C : ℝ)

-- Define the new tax rate after 20% decrease
def new_tax (T : ℝ) : ℝ := 0.8 * T

-- Define the new consumption after 20% increase
def new_consumption (C : ℝ) : ℝ := 1.2 * C

-- Define the original revenue
def original_revenue (T C : ℝ) : ℝ := T * C

-- Define the new revenue
def new_revenue (T C : ℝ) : ℝ := new_tax T * new_consumption C

-- Theorem statement
theorem revenue_decrease_percent (T C : ℝ) :
  (original_revenue T C - new_revenue T C) / original_revenue T C * 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_decrease_percent_l497_49729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_lines_intersect_parabola_l497_49756

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define a line passing through (0,4)
def line_through_0_4 (k b : ℝ) (x y : ℝ) : Prop :=
  y = k*x + b ∧ 4 = b

-- Define the condition for a line to intersect the parabola at exactly one point
def intersects_once (k b : ℝ) : Prop :=
  ∃! x : ℝ, ∃ y : ℝ, line_through_0_4 k b x y ∧ parabola x y

-- The main theorem
theorem three_lines_intersect_parabola :
  ∃! (lines : Finset (ℝ × ℝ)), 
    lines.card = 3 ∧ 
    (∀ l ∈ lines, let (k, b) := l; intersects_once k b) ∧
    (∀ k b : ℝ, intersects_once k b → (k, b) ∈ lines) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_lines_intersect_parabola_l497_49756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_perimeter_l497_49739

/-- Represents a right triangle -/
def IsRightTriangle : (Set ℝ × Set ℝ) → Prop := sorry

/-- Represents a line segment in the real plane -/
structure RealSegment where
  length : ℝ

/-- Represents an angle -/
structure Angle where
  value : ℝ

/-- Calculates the area of a triangle -/
def triangleArea : Set ℝ × Set ℝ → ℝ := sorry

/-- Calculates the perimeter of a triangle -/
def trianglePerimeter : Set ℝ × Set ℝ → ℝ := sorry

/-- 
Given a right triangle with hypotenuse 10√2 inches and one angle of 45°, 
prove that its area is 50 square inches and its perimeter is 20 + 10√2 inches.
-/
theorem right_triangle_area_perimeter 
  (triangle : Set ℝ × Set ℝ) 
  (is_right_triangle : IsRightTriangle triangle) 
  (hypotenuse : RealSegment) 
  (angle : Angle) :
  (hypotenuse.length = 10 * Real.sqrt 2) →
  (angle.value = π / 4) →
  (triangleArea triangle = 50) ∧ 
  (trianglePerimeter triangle = 20 + 10 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_perimeter_l497_49739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_points_finite_not_computable_l497_49778

-- Define a structure for rational numbers with prime numerator and denominator
structure PrimeRational where
  num : Nat
  den : Nat
  num_prime : Nat.Prime num
  den_prime : Nat.Prime den
  pos : 0 < num ∧ 0 < den

-- Define the set of points satisfying the conditions
def ValidPoints : Set (PrimeRational × PrimeRational) :=
  {p | (p.1.num : ℚ) / p.1.den + (p.2.num : ℚ) / p.2.den ≤ 7}

-- State the theorem
theorem valid_points_finite_not_computable :
  (Set.Finite ValidPoints) ∧
  ¬∃ (n : Nat), ∃ (h : Fintype ValidPoints), Fintype.card ValidPoints = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_points_finite_not_computable_l497_49778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_a_range_l497_49736

-- Define the ellipse E
noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line (x y : ℝ) : Prop :=
  x + 2*y - 2 = 0

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 2 / 2

-- Theorem for part (Ⅰ)
theorem ellipse_equation :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  ellipse a b 2 0 ∧
  (∀ x y, ellipse a b x y ↔ x^2/4 + y^2/2 = 1) := by
  sorry

-- Theorem for part (Ⅱ)
theorem a_range :
  ∀ (a b : ℝ), a > b ∧ b > 0 →
  (∃ (x y : ℝ), line x y ∧ 0 ≤ x ∧ x ≤ 2 ∧
    (∃ (f1x f1y f2x f2y : ℝ),
      ellipse a b f1x f1y ∧ ellipse a b f2x f2y ∧
      Real.sqrt ((x - f1x)^2 + (y - f1y)^2) +
      Real.sqrt ((x - f2x)^2 + (y - f2y)^2) = 2*a)) →
  2 * Real.sqrt 3 / 3 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_a_range_l497_49736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_solution_l497_49786

/-- The smallest positive angle in degrees that satisfies the given trigonometric equation -/
noncomputable def smallest_angle : ℝ := 180 / 7

/-- The trigonometric equation to be satisfied -/
def trig_equation (x : ℝ) : Prop :=
  Real.tan (6 * x * Real.pi / 180) = (Real.sin (x * Real.pi / 180) - Real.cos (x * Real.pi / 180)) / 
                                     (Real.sin (x * Real.pi / 180) + Real.cos (x * Real.pi / 180))

theorem smallest_angle_solution :
  (∀ y, 0 < y ∧ y < smallest_angle → ¬ trig_equation y) ∧
  trig_equation smallest_angle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_solution_l497_49786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l497_49785

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

/-- The theorem statement -/
theorem geometric_sequence_sum_five (q : ℝ) :
  q > 0 ∧ 
  geometric_sum 1 q 4 - 5 * geometric_sum 1 q 2 = 0 →
  geometric_sum 1 q 5 = 31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l497_49785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_series_convergence_l497_49732

/-- The double series ∑_{m=1}^∞ ∑_{n=m}^∞ 1/(mn(m+n+2)) converges to π²/6 -/
theorem double_series_convergence :
  let f : ℕ+ → ℕ+ → ℝ := λ m n => (1 : ℝ) / (m * n * (m + n + 2))
  (∑' m : ℕ+, ∑' n : ℕ+, if n ≥ m then f m n else 0) = π^2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_series_convergence_l497_49732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_l497_49789

/-- A quadratic function with real coefficients -/
def quadratic_function (a b : ℝ) : ℝ → ℝ := λ x ↦ x^2 + a*x + b

/-- The property that the range of a function is [0, +∞) -/
def nonnegative_range (f : ℝ → ℝ) : Prop :=
  ∀ y, (∃ x, f x = y) → y ≥ 0

/-- The solution set of an inequality f(x) < c is an open interval (m, m+8) -/
def solution_set_is_open_interval (f : ℝ → ℝ) (c m : ℝ) : Prop :=
  ∀ x, f x < c ↔ m < x ∧ x < m + 8

theorem quadratic_inequality_solution (a b c m : ℝ) :
  nonnegative_range (quadratic_function a b) →
  solution_set_is_open_interval (quadratic_function a b) c m →
  c = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_l497_49789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l497_49781

-- Define the nabla operation as noncomputable
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- State the theorem
theorem nabla_calculation :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
  nabla (nabla a b) c = 11 / 9 :=
by
  -- The proof is not provided, so we use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l497_49781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l497_49709

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola -/
structure Parabola where
  vertex : Point
  focus : Point
  opensDownward : Bool

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Check if a point lies on a parabola -/
def isOnParabola (p : Point) (par : Parabola) : Prop :=
  if par.opensDownward then
    p.x^2 = -4 * (par.focus.y - par.vertex.y) * (p.y - par.vertex.y)
  else
    p.x^2 = 4 * (par.focus.y - par.vertex.y) * (p.y - par.vertex.y)

theorem parabola_point_coordinates :
  ∀ (par : Parabola) (p : Point),
    par.vertex = Point.mk 0 0 →
    par.focus = Point.mk 0 (-1) →
    par.opensDownward →
    isOnParabola p par →
    isInFourthQuadrant p →
    distance p par.focus = 121 →
    p = Point.mk (20 * Real.sqrt 6) (-120) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l497_49709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_twin_primes_disproves_conjecture_l497_49717

-- Define twin primes
def is_twin_prime (p : ℕ) : Prop := Nat.Prime p ∧ Nat.Prime (p + 2)

-- Define the set of all twin primes
def twin_prime_set : Set ℕ := {p | is_twin_prime p}

-- The Twin Prime Conjecture
def twin_prime_conjecture : Prop := Set.Infinite twin_prime_set

-- Theorem statement
theorem finite_twin_primes_disproves_conjecture :
  Set.Finite twin_prime_set → ¬twin_prime_conjecture := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_twin_primes_disproves_conjecture_l497_49717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_solution_l497_49716

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -2; -2, -1]
def B : Matrix (Fin 2) (Fin 1) ℝ := !![5; -15]
def X : Matrix (Fin 2) (Fin 1) ℝ := !![7; 1]

theorem matrix_equation_solution :
  A * X = B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_solution_l497_49716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_winning_n_l497_49782

/-- Represents the game board --/
def Board := Fin 2018

/-- Represents a valid move in the game --/
def ValidMove (n : ℕ) := Fin n

/-- Represents the game state --/
structure GameState (n : ℕ) where
  position : Board
  moves : List (ValidMove n)

/-- Defines a position where Jerry cannot move --/
def IsWinningPosition (n : ℕ) (position : Board) :=
  (position.val ≤ n ∧ position.val > 2018 - n)

/-- Defines a winning strategy for Tom --/
def WinningStrategy (n : ℕ) :=
  ∀ (initial : Board), ∃ (moves : List (ValidMove n)), 
    (∀ (jerry_moves : List Bool), 
      (jerry_moves.length = moves.length) → 
      ∃ (final : Board), IsWinningPosition n final)

/-- The main theorem to be proved --/
theorem smallest_winning_n : 
  (∀ m < 1010, ¬ WinningStrategy m) ∧ WinningStrategy 1010 := by
  sorry

/-- Auxiliary lemma: For any n ≤ 1009, Jerry can always move --/
lemma jerry_can_always_move (n : ℕ) (h : n ≤ 1009) :
  ∀ (position : Board), ¬IsWinningPosition n position := by
  sorry

/-- Auxiliary lemma: Tom has a winning strategy for n = 1010 --/
lemma tom_wins_at_1010 : WinningStrategy 1010 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_winning_n_l497_49782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l497_49795

noncomputable section

def f (x : ℝ) := Real.log x - Real.exp (1 - x)
def g (a : ℝ) (x : ℝ) := a * (x^2 - 1) - 1 / x
def h (a : ℝ) (x : ℝ) := g a x - f x + (Real.exp x - Real.exp 1 * x) / (x * Real.exp x)

theorem problem_solution :
  (∃! x, x > 0 ∧ f x = 0) ∧
  (∀ a ≤ 0, StrictMono (fun x => -h a x)) ∧
  (∀ a > 0, ∃ x₀ > 0, (∀ x ∈ Set.Ioo 0 x₀, StrictMono (fun x => -h a x)) ∧
                       (∀ x ∈ Set.Ioi x₀, StrictMono (h a))) ∧
  (∀ x > 1, f x < g (1/2) x) ∧
  (∀ a ≥ 1/2, ∀ x > 1, f x < g a x) ∧
  (∀ a < 1/2, ∃ x > 1, f x ≥ g a x) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l497_49795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_minus_constant_l497_49726

theorem sum_of_coefficients_minus_constant :
  let f : ℝ → ℝ := fun x ↦ (1 - 2*x)^7
  let g : ℝ → ℝ := fun x ↦ a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7
  f = g →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 :=
by
  intros f g h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_minus_constant_l497_49726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_earnings_theorem_l497_49753

/-- Represents the earnings of an Italian restaurant for a month. -/
structure RestaurantEarnings where
  weekdayEarnings : ℕ  -- Regular weekday earnings in cents
  weekendEarningsLower : ℕ  -- Lower bound of weekend earnings in cents
  weekendEarningsUpper : ℕ  -- Upper bound of weekend earnings in cents
  mondayDiscount : ℕ  -- Monday discount percentage
  specialEventEarnings : ℕ  -- Special event earnings in cents
  weekdayCount : ℕ  -- Number of weekdays in the month
  weekendCount : ℕ  -- Number of weekends in the month

/-- Calculates the total monthly earnings for the restaurant. -/
def calculateTotalEarnings (r : RestaurantEarnings) : ℕ := 
  let avgWeekendEarnings := (r.weekendEarningsLower + r.weekendEarningsUpper) / 2
  let mondayCount := r.weekdayCount / 5  -- Approximate number of Mondays
  let mondayEarnings := mondayCount * r.weekdayEarnings * (100 - r.mondayDiscount) / 100
  let otherWeekdayEarnings := (r.weekdayCount - mondayCount) * r.weekdayEarnings
  let weekendEarnings := r.weekendCount * 2 * avgWeekendEarnings
  mondayEarnings + otherWeekdayEarnings + weekendEarnings + r.specialEventEarnings

/-- Theorem stating that the restaurant's total monthly earnings are $33,460. -/
theorem restaurant_earnings_theorem (r : RestaurantEarnings) 
  (h1 : r.weekdayEarnings = 60000)
  (h2 : r.weekendEarningsLower = 100000)
  (h3 : r.weekendEarningsUpper = 150000)
  (h4 : r.mondayDiscount = 10)
  (h5 : r.specialEventEarnings = 50000)
  (h6 : r.weekdayCount = 22)
  (h7 : r.weekendCount = 8) :
  calculateTotalEarnings r = 3346000 := by
  sorry

#eval calculateTotalEarnings {
  weekdayEarnings := 60000,
  weekendEarningsLower := 100000,
  weekendEarningsUpper := 150000,
  mondayDiscount := 10,
  specialEventEarnings := 50000,
  weekdayCount := 22,
  weekendCount := 8
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_earnings_theorem_l497_49753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sum_partition_l497_49710

def A : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 72) (Finset.range 73)

theorem equal_sum_partition :
  ∃ B : Finset ℕ, B ⊆ A ∧ B.card = 36 ∧
  Finset.sum (B) id = Finset.sum (A \ B) id :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sum_partition_l497_49710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_points_cyclic_l497_49724

noncomputable section

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Four circles where adjacent circles touch externally -/
structure FourTouchingCircles where
  S₁ : Circle
  S₂ : Circle
  S₃ : Circle
  S₄ : Circle
  touch_12 : (S₁.center.1 - S₂.center.1)^2 + (S₁.center.2 - S₂.center.2)^2 = (S₁.radius + S₂.radius)^2
  touch_23 : (S₂.center.1 - S₃.center.1)^2 + (S₂.center.2 - S₃.center.2)^2 = (S₂.radius + S₃.radius)^2
  touch_34 : (S₃.center.1 - S₄.center.1)^2 + (S₃.center.2 - S₄.center.2)^2 = (S₃.radius + S₄.radius)^2
  touch_41 : (S₄.center.1 - S₁.center.1)^2 + (S₄.center.2 - S₁.center.2)^2 = (S₄.radius + S₁.radius)^2

/-- The point of tangency between two circles -/
def pointOfTangency (C₁ C₂ : Circle) : ℝ × ℝ :=
  let t := C₁.radius / (C₁.radius + C₂.radius)
  (t * C₂.center.1 + (1 - t) * C₁.center.1, t * C₂.center.2 + (1 - t) * C₁.center.2)

/-- Four points are cyclic if they lie on the same circle -/
def areCyclic (A B C D : ℝ × ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
    (B.1 - center.1)^2 + (B.2 - center.2)^2 = radius^2 ∧
    (C.1 - center.1)^2 + (C.2 - center.2)^2 = radius^2 ∧
    (D.1 - center.1)^2 + (D.2 - center.2)^2 = radius^2

theorem tangency_points_cyclic (circles : FourTouchingCircles) :
  let A := pointOfTangency circles.S₁ circles.S₂
  let B := pointOfTangency circles.S₂ circles.S₃
  let C := pointOfTangency circles.S₃ circles.S₄
  let D := pointOfTangency circles.S₄ circles.S₁
  areCyclic A B C D := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_points_cyclic_l497_49724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l497_49728

/-- Given two vectors a and b in R², where a = (2,x) and b = (-3,2),
    if a + b is perpendicular to b, then x = -7/2 -/
theorem perpendicular_vectors (x : ℝ) : 
  let a : ℝ × ℝ := (2, x)
  let b : ℝ × ℝ := (-3, 2)
  (a.1 + b.1, a.2 + b.2) • b = 0 → x = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l497_49728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_primes_in_examples_l497_49771

/-- A type representing an arithmetic example of the form a + b = c -/
structure ArithmeticExample where
  a : ℕ
  b : ℕ
  c : ℕ
  sum_eq : a + b = c

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Bool := Nat.Prime n

/-- A function that counts the number of prime numbers in a list of natural numbers -/
def countPrimes (numbers : List ℕ) : ℕ :=
  numbers.filter isPrime |>.length

theorem max_primes_in_examples (examples : List ArithmeticExample) :
  examples.length = 20 →
  (examples.map (λ e => [e.a, e.b, e.c])).join.toFinset.card = 60 →
  (∀ e ∈ examples, e.a ≠ e.b ∧ e.b ≠ e.c ∧ e.a ≠ e.c) →
  countPrimes ((examples.map (λ e => [e.a, e.b, e.c])).join) ≤ 41 :=
by sorry

#check max_primes_in_examples

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_primes_in_examples_l497_49771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rational_roots_l497_49741

def polynomial (n : ℕ) (x : ℚ) : ℚ := 
  Finset.sum (Finset.range (n+1)) (λ k ↦ x^k / Nat.factorial k)

theorem no_rational_roots (n : ℕ) (h : n > 1) : 
  ¬ ∃ (x : ℚ), polynomial n x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rational_roots_l497_49741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_deriv_neg_l497_49703

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivatives of f and g
variable (f' g' : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x
axiom f_deriv_pos : ∀ x, x > 0 → HasDerivAt f (f' x) x ∧ f' x > 0
axiom g_deriv_pos : ∀ x, x > 0 → HasDerivAt g (g' x) x ∧ g' x > 0

-- State the theorem
theorem f_g_deriv_neg (x : ℝ) (hx : x < 0) : f' x > 0 ∧ g' x < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_deriv_neg_l497_49703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quartic_polynomial_root_l497_49714

theorem monic_quartic_polynomial_root (x : ℝ) : x = (3 : ℝ) ^ (1/4) + 1 →
  x^4 - 4*x^3 + 6*x^2 - 4*x - 2 = 0 ∧
  (∃ (a b c : ℤ), x^4 - 4*x^3 + 6*x^2 - 4*x - 2 = x^4 + a*x^3 + b*x^2 + c*x - 2) :=
by sorry

#check monic_quartic_polynomial_root

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quartic_polynomial_root_l497_49714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_shortest_path_unique_line_through_points_l497_49742

-- Define a point in a 2D plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a line segment between two points
def LineSegment (a b : Point) := {p : Point | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p.x = (1 - t) * a.x + t * b.x ∧ p.y = (1 - t) * a.y + t * b.y}

-- Define a straight line passing through two points
def StraightLine (a b : Point) := {p : Point | ∃ t : ℝ, p.x = (1 - t) * a.x + t * b.x ∧ p.y = (1 - t) * a.y + t * b.y}

-- Define the distance between two points
noncomputable def distance (a b : Point) : ℝ := Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

-- Theorem 1: The line segment is the shortest path between two points
theorem segment_shortest_path (a b : Point) (p : Point) 
  (h : p ∈ StraightLine a b) (h_neq : p ≠ a ∧ p ≠ b) : 
  distance a p + distance p b ≥ distance a b := by
  sorry

-- Theorem 2: There exists a unique straight line passing through any two distinct points
theorem unique_line_through_points (a b : Point) (h : a ≠ b) :
  ∃! l : Set Point, l = StraightLine a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_shortest_path_unique_line_through_points_l497_49742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_zero_l497_49759

theorem derivative_at_zero (f : ℝ → ℝ) :
  (∀ x, f x = Real.exp x + 2 * x * (deriv f 1)) →
  deriv f 0 = 1 - 2 * Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_zero_l497_49759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_distance_theorem_l497_49701

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def Line.passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

noncomputable def distance (l1 l2 : Line) : ℝ :=
  |l1.c - l2.c| / Real.sqrt (l1.a^2 + l1.b^2)

theorem line_and_distance_theorem (l1 : Line) (P : Point) :
  l1.a = 3 ∧ l1.b = 4 ∧ l1.c = -12 ∧ P.x = 4 ∧ P.y = -5 →
  ∃ l2 : Line,
    l2.a = 3 ∧ l2.b = 4 ∧ l2.c = 8 ∧
    parallel l1 l2 ∧
    l2.passes_through P ∧
    distance l1 l2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_distance_theorem_l497_49701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_effective_expected_cycles_harmonic_l497_49772

/-- Represents a system of mailboxes with randomly placed keys -/
structure MailboxSystem (n : ℕ) where
  key_placement : Fin n → Fin n

/-- The algorithm for finding one's key -/
def find_key (system : MailboxSystem n) (start : Fin n) : List (Fin n) :=
  sorry

/-- Theorem: The algorithm always terminates with the correct key -/
theorem algorithm_effective (n : ℕ) (system : MailboxSystem n) (start : Fin n) :
  ∃ (k : ℕ), (find_key system start).get? k = some start :=
  sorry

/-- Expected number of cycles in the graph representation -/
noncomputable def expected_cycles (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k => 1 / (k + 1 : ℝ))

/-- Theorem: The expected number of cycles equals the nth harmonic number -/
theorem expected_cycles_harmonic (n : ℕ) :
  expected_cycles n = (Finset.range n).sum (λ k => 1 / (k + 1 : ℝ)) :=
by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_effective_expected_cycles_harmonic_l497_49772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_for_cos_three_fourths_l497_49702

theorem angle_range_for_cos_three_fourths (A : ℝ) (h : Real.cos A = 3/4) : 
  30 * (Real.pi / 180) < A ∧ A < 45 * (Real.pi / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_for_cos_three_fourths_l497_49702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_condition_sequence_existence_characterization_l497_49752

open Set
open Real

theorem sequence_existence_condition (l : ℝ) : Prop :=
  ¬∃ (a : ℕ → ℝ), (∀ n, n ≥ 2 → a n > 0) ∧ 
    (∀ n, n ≥ 2 → a n + 1 ≤ (l ^ (1 / n : ℝ)) * a (n - 1))

theorem sequence_existence_characterization :
  ∀ l : ℝ, sequence_existence_condition l ↔ l ∈ Ioc 0 (Real.exp 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_condition_sequence_existence_characterization_l497_49752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_safe_time_l497_49758

noncomputable def drug_concentration (t : ℝ) : ℝ :=
  if 0 ≤ t ∧ t ≤ 0.1 then 10 * t
  else (1 / 16) ^ (t - 1 / 10)

theorem minimum_safe_time :
  ∀ t : ℝ, t ≥ 0 → (drug_concentration t ≤ 0.25 ↔ t ≥ 0.6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_safe_time_l497_49758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_equals_combinations_l497_49725

theorem solutions_count_equals_combinations (n : ℕ) :
  (Nat.choose (n + 3 - 1) 2) = 
  Finset.card (Finset.filter (fun tuple : ℕ × ℕ × ℕ => tuple.1 + tuple.2.1 + tuple.2.2 = n) (Finset.product (Finset.range (n + 1)) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1))))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_equals_combinations_l497_49725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_and_f_extrema_l497_49773

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (3*x/2), Real.sin (3*x/2))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (x/2), -Real.sin (x/2))

noncomputable def f (x : ℝ) : ℝ := 
  (a x).1 * (b x).1 + (a x).2 * (b x).2 - Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2

theorem vector_sum_magnitude_and_f_extrema :
  (∀ x ∈ Set.Icc (-π/3) (π/4), 
    Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2 = 2 * Real.cos x) ∧
  (∃ x₁ ∈ Set.Icc (-π/3) (π/4), ∀ x ∈ Set.Icc (-π/3) (π/4), f x ≤ f x₁) ∧
  (∃ x₁ ∈ Set.Icc (-π/3) (π/4), f x₁ = -1) ∧
  (∃ x₂ ∈ Set.Icc (-π/3) (π/4), ∀ x ∈ Set.Icc (-π/3) (π/4), f x ≥ f x₂) ∧
  (∃ x₂ ∈ Set.Icc (-π/3) (π/4), f x₂ = -Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_and_f_extrema_l497_49773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_jump_impossible_l497_49731

-- Define the value of ω
noncomputable def ω : ℝ := (-1 + Real.sqrt 5) / 2

-- Define the value function for a point (x, y)
noncomputable def value (x y : ℤ) : ℝ := ω ^ (Int.natAbs x - y)

-- Define the sum of values for all points in the lower half-plane
noncomputable def lower_half_plane_sum : ℝ := 1 / ((1 - ω) ^ 3)

-- Define a valid move (this is a placeholder and needs to be properly defined)
def valid_move (s1 s2 : Set (ℤ × ℤ)) : Prop := sorry

-- Theorem statement
theorem chess_jump_impossible (pieces : Set (ℤ × ℤ)) 
  (h_initial : ∀ (x y : ℤ), (x, y) ∈ pieces → y ≤ 0 ∨ y ≥ 0) :
  ¬ ∃ (moves : ℕ → Set (ℤ × ℤ)), 
    (moves 0 = pieces) ∧ 
    (∀ n : ℕ, valid_move (moves n) (moves (n+1))) ∧
    (∃ (n : ℕ) (x : ℤ), (x, 5) ∈ moves n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_jump_impossible_l497_49731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_quitters_same_group_l497_49792

/-- The probability of all three quitters coming from the same group -/
theorem probability_three_quitters_same_group :
  let total_participants : ℕ := 20
  let group_size : ℕ := 10
  let num_quitters : ℕ := 3
  let total_combinations := Nat.choose total_participants num_quitters
  let same_group_combinations := 2 * Nat.choose group_size num_quitters
  (same_group_combinations : ℚ) / total_combinations = 20 / 95 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_quitters_same_group_l497_49792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l497_49722

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x - (1/2) * x

-- State the theorem
theorem range_of_f :
  ∃ (a b : ℝ), a = 0 ∧ b = (3 * Real.sqrt 3 - Real.pi) / 6 ∧
  (∀ y : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = y) ↔ y ∈ Set.Icc a b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l497_49722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l497_49774

/-- The y-intercept of a line is the y-coordinate of the point where the line crosses the y-axis (i.e., where x = 0) -/
noncomputable def y_intercept (a b c : ℝ) : ℝ := -c / b

/-- A line is defined by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem y_intercept_of_line :
  let l : Line := { a := 2, b := -3, c := -6 }
  y_intercept l.a l.b l.c = -2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l497_49774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_g_l497_49735

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem solution_set_of_g (f : ℝ → ℝ) (h_odd : IsOdd f) 
  (h_deriv : ∀ x > 0, x * (deriv f) x > 2 * f (-x)) :
  let g := fun x ↦ x^2 * f x
  ∀ x, g x < g (1 - 3*x) ↔ x < 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_g_l497_49735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_price_is_two_l497_49799

/-- Represents the price reduction factor -/
noncomputable def reduction_factor : ℝ := 0.7

/-- Represents the additional number of apples that can be bought after the price reduction -/
def additional_apples : ℕ := 54

/-- Represents the fixed amount of money spent on apples -/
noncomputable def fixed_amount : ℝ := 30

/-- Calculates the reduced price per dozen apples -/
noncomputable def reduced_price_per_dozen (original_price : ℝ) : ℝ :=
  let reduced_price := original_price * reduction_factor
  let original_apples := fixed_amount / original_price
  let new_apples := original_apples + additional_apples
  let price_per_apple := fixed_amount / new_apples
  12 * price_per_apple

/-- Theorem stating that the reduced price per dozen apples is 2.00 -/
theorem reduced_price_is_two : ∃ (original_price : ℝ), reduced_price_per_dozen original_price = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_price_is_two_l497_49799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salmon_weight_calculation_l497_49747

/-- The weight of each salmon in pounds -/
def salmon_weight : ℝ := 12

/-- The number of salmon caught -/
def num_salmon : ℕ := 2

/-- The weight of the trout in pounds -/
def trout_weight : ℝ := 8

/-- The number of bass caught -/
def num_bass : ℕ := 6

/-- The weight of each bass in pounds -/
def bass_weight : ℝ := 2

/-- The number of campers -/
def num_campers : ℕ := 22

/-- The amount of fish needed per camper in pounds -/
def fish_per_camper : ℝ := 2

theorem salmon_weight_calculation :
  salmon_weight * (num_salmon : ℝ) + trout_weight + (num_bass : ℝ) * bass_weight =
  (num_campers : ℝ) * fish_per_camper := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salmon_weight_calculation_l497_49747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_f_range_l497_49769

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (x + 2) * Real.exp (-x) - 2
  else (x - 2) * Real.exp x + 2

-- Theorem stating that f is an odd function
theorem f_odd : ∀ x, f (-x) = -f x := by sorry

-- Theorem for the range of f(x) when x ∈ [0, 2]
theorem f_range : Set.range (fun x ↦ f x) ∩ Set.Icc 0 2 = Set.Icc (2 - Real.exp 1) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_f_range_l497_49769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l497_49783

theorem cube_root_inequality (x : ℝ) : 
  x ^ (1/3) + 1 / (x ^ (1/3) + 2) ≤ 0 ↔ -8 < x ∧ x ≤ -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l497_49783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_abs_value_l497_49713

-- Define the integral
noncomputable def integral (f : ℝ → ℝ) (a b : ℝ) : ℝ := ∫ x in a..b, f x

-- Define the absolute value function
def abs_val (x : ℝ) : ℝ := |x|

-- Define the exponential function
noncomputable def exp_func (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem integral_exp_abs_value :
  integral (λ x ↦ exp_func (abs_val x)) (-2) 4 = exp_func 4 + exp_func 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_abs_value_l497_49713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_takeoff_run_length_theorem_l497_49787

/-- Represents the properties of an airplane during takeoff -/
structure Airplane where
  takeoff_time : ℝ
  takeoff_speed : ℝ
  uniformly_accelerated : Prop
  starts_from_rest : Prop

/-- Calculates the length of the takeoff run for a given airplane -/
noncomputable def takeoff_run_length (a : Airplane) : ℝ :=
  let acceleration := a.takeoff_speed / a.takeoff_time
  (1/2) * acceleration * a.takeoff_time^2

/-- Theorem stating the length of the takeoff run for a specific airplane -/
theorem takeoff_run_length_theorem (a : Airplane) 
  (h1 : a.takeoff_time = 15)
  (h2 : a.takeoff_speed = 100 * 1000 / 3600)
  (h3 : a.uniformly_accelerated)
  (h4 : a.starts_from_rest) :
  ∃ (ε : ℝ), abs (takeoff_run_length a - 208) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_takeoff_run_length_theorem_l497_49787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l497_49749

noncomputable def f (x : ℝ) := 2 + 3*x + 4/(x - 1)

theorem min_value_of_f :
  ∃ (x_min : ℝ), x_min > 1 ∧
    (∀ (x : ℝ), x > 1 → f x ≥ f x_min) ∧
    f x_min = 5 + 4 * Real.sqrt 3 ∧
    x_min = 1 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l497_49749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_length_l497_49793

/-- The length of the diagonal of a square with area 7.22 square meters -/
noncomputable def diagonal_length : ℝ := Real.sqrt (2 * 7.22)

/-- Theorem: The length of the diagonal of a square with area 7.22 square meters
    is equal to √(2 * 7.22) meters. -/
theorem square_diagonal_length :
  ∀ (s : ℝ), s > 0 → s^2 = 7.22 → Real.sqrt (2 * s^2) = diagonal_length :=
by
  sorry

-- Using #eval with Real.sqrt may not work as expected in Lean 4
-- Instead, we can use a floating-point approximation
#eval Float.sqrt (2 * 7.22)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_length_l497_49793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sail_time_difference_l497_49798

/-- Represents the wind speed as a function of time -/
noncomputable def wind_speed (t : ℝ) : ℝ := t / 10

/-- Calculates the average speed for a given sail size and wind speed -/
noncomputable def avg_speed (sail_size : ℝ) (w : ℝ) : ℝ :=
  if sail_size = 24 then 50 * w else 20 * w

/-- Calculates the time taken to travel a given distance at a given average speed -/
noncomputable def travel_time (distance : ℝ) (avg_speed : ℝ) : ℝ := distance / avg_speed

/-- The main theorem stating the time difference between using two different sail sizes -/
theorem sail_time_difference :
  let big_sail := 24
  let small_sail := 12
  let total_distance := 200
  let avg_wind_speed := (wind_speed 0 + wind_speed 10) / 2
  travel_time total_distance (avg_speed small_sail avg_wind_speed) -
  travel_time total_distance (avg_speed big_sail avg_wind_speed) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sail_time_difference_l497_49798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_obtuse_triangle_approx_num_pairs_is_six_probability_relationship_l497_49754

/-- The probability that no two points form an obtuse triangle with the circle's center
    when four points are chosen uniformly at random on a circle -/
noncomputable def probability_no_obtuse_triangle : ℝ := (3/8)^6

/-- Four points chosen uniformly at random on a circle -/
def random_points : ℕ := 4

/-- Theorem stating the probability of no obtuse triangles formed with the circle's center -/
theorem probability_no_obtuse_triangle_approx :
  probability_no_obtuse_triangle = (3/8)^6 :=
by sorry

/-- The number of pairs that can be formed from the random points -/
def num_pairs : ℕ := Nat.choose random_points 2

/-- Theorem stating the number of pairs is 6 -/
theorem num_pairs_is_six : num_pairs = 6 :=
by sorry

/-- The probability that the minor arc between any two points is less than π/2 -/
noncomputable def prob_minor_arc_less_than_half_pi : ℝ := 1/2

/-- Theorem stating the relationship between the probability and the number of pairs -/
theorem probability_relationship :
  probability_no_obtuse_triangle = prob_minor_arc_less_than_half_pi ^ num_pairs :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_obtuse_triangle_approx_num_pairs_is_six_probability_relationship_l497_49754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_eight_leftmost_digit_l497_49797

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The leftmost digit of a positive integer -/
def leftmost_digit (n : ℕ) : ℕ := sorry

/-- The set S of powers of 8 -/
def S : Set ℕ := {n | ∃ k : ℕ, 0 ≤ k ∧ k ≤ 3000 ∧ n = 8^k}

theorem unique_eight_leftmost_digit :
  (num_digits (8^3000) = 2857) →
  (leftmost_digit (8^3000) = 8) →
  ∃! k : ℕ, k ∈ S ∧ leftmost_digit k = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_eight_leftmost_digit_l497_49797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_to_line_distance_l497_49723

theorem point_to_line_distance (θ : Real) :
  0 ≤ θ ∧ θ ≤ Real.pi / 2 →
  let point := (Real.cos θ, Real.sin θ)
  let line := fun (x y : Real) ↦ x * Real.sin θ + y * Real.cos θ - 1 = 0
  let distance := |Real.sin θ * Real.cos θ + Real.sin θ * Real.cos θ - 1| / Real.sqrt (Real.cos θ ^ 2 + Real.sin θ ^ 2)
  distance = 1 / 2 →
  θ = Real.pi / 12 ∨ θ = 5 * Real.pi / 12 := by
  sorry

#check point_to_line_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_to_line_distance_l497_49723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pension_calculation_l497_49706

theorem pension_calculation (a b p q : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  let pension (x : ℝ) := k * x^2
  let k : ℝ := (pension (x + 2*a) - pension x) / (2*p)
  ∀ x : ℝ, 
    pension (x + 2*a) - pension x = 2*p ∧
    pension (x + 3*b) - pension x = 3*q →
    pension x = -(b*p - 2*a*q) / (6*a*b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pension_calculation_l497_49706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_plus_x_l497_49746

theorem integral_sqrt_one_minus_x_squared_plus_x :
  (∫ x in Set.Icc 0 1, (Real.sqrt (1 - x^2) + x)) = π / 4 + 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_plus_x_l497_49746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_modulus_for_67_power_plus_67_l497_49764

theorem unique_modulus_for_67_power_plus_67 :
  ∃! n : ℕ, n > 0 ∧ (67^67 + 67) % n = 66 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_modulus_for_67_power_plus_67_l497_49764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l497_49755

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 + x - 6 ≤ 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x > 1}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l497_49755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l497_49765

theorem undefined_values_count : ∃! (S : Finset ℝ), 
  (∀ x ∈ S, (x^2 + 3*x - 4)*(x - 4) = 0) ∧ 
  (∀ x ∉ S, (x^2 + 3*x - 4)*(x - 4) ≠ 0) ∧ 
  S.card = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l497_49765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_l497_49730

/-- A set of points in ℝ² forms a regular hexagon. -/
class IsRegularHexagon (S : Set (ℝ × ℝ)) : Prop

/-- The area of a set of points in ℝ². -/
noncomputable def Set.area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The area of a regular hexagon with vertices P at (0,0) and R at (8,2) is 102√3. -/
theorem regular_hexagon_area : ∃ (PQRSTU : Set (ℝ × ℝ)),
  IsRegularHexagon PQRSTU ∧
  (0, 0) ∈ PQRSTU ∧
  (8, 2) ∈ PQRSTU ∧
  Set.area PQRSTU = 102 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_l497_49730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l497_49757

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (3 * x - Real.pi / 4) + 1

theorem sine_function_properties :
  ∃ (phase_shift vertical_shift max_value min_value : ℝ),
    phase_shift = Real.pi / 12 ∧
    vertical_shift = 1 ∧
    max_value = 4 ∧
    min_value = -2 ∧
    (∀ x, f x ≤ max_value) ∧
    (∀ x, f x ≥ min_value) ∧
    (∀ x, f (x + phase_shift) = f x - vertical_shift) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l497_49757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_guessing_game_bounds_l497_49796

/-- Represents the response given by Player A to a guess -/
inductive Response
| Hot
| Cold

/-- Checks if a number is a valid two-digit number -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Determines if a guess should receive a "hot" response -/
def is_hot_response (target : ℕ) (guess : ℕ) : Prop :=
  is_two_digit target ∧ is_two_digit guess ∧
  (target = guess ∨ 
   (target / 10 = guess / 10 ∧ Int.natAbs (target % 10 - guess % 10) ≤ 1) ∨
   (target % 10 = guess % 10 ∧ Int.natAbs (target / 10 - guess / 10) ≤ 1))

/-- The main theorem stating the bounds on the number of guesses required -/
theorem guessing_game_bounds :
  ∀ (target : ℕ), is_two_digit target →
  (∃ (strategy : ℕ → ℕ), ∀ (n : ℕ), n ≤ 22 → 
    (is_hot_response target (strategy n) → target = strategy n)) ∧
  (∃ (guesses : List ℕ), guesses.length = 18 ∧
    (∀ guess ∈ guesses, ¬is_hot_response target guess) ∧
    ∃ (remaining : ℕ), is_two_digit remaining ∧ remaining ∉ guesses) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_guessing_game_bounds_l497_49796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_s_squared_l497_49745

/-- A hyperbola centered at (0,2) passing through (-1, 6), (0, 5), and (s, 3) -/
structure Hyperbola where
  -- The hyperbola passes through (-1, 6)
  passes_through_neg1_6 : ((-1:ℝ)^2 / a^2) - ((6:ℝ)-2)^2 / b^2 = 1
  -- The hyperbola passes through (0, 5)
  passes_through_0_5 : ((0:ℝ)^2 / a^2) - ((5:ℝ)-2)^2 / b^2 = 1
  -- The hyperbola passes through (s, 3)
  passes_through_s_3 : (s^2 / a^2) - ((3:ℝ)-2)^2 / b^2 = 1
  -- The hyperbola is centered at (0,2)
  center : ℝ × ℝ := (0, 2)
  a : ℝ
  b : ℝ
  s : ℝ

/-- The theorem stating that s^2 = 2/5 for the given hyperbola -/
theorem hyperbola_s_squared (h : Hyperbola) : h.s^2 = 2/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_s_squared_l497_49745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_shifted_arctan_l497_49780

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.arctan x

-- Define the shifted function (graph C)
noncomputable def g (x : ℝ) : ℝ := f (x - 2)

-- Define the symmetric function (graph C')
noncomputable def h (x : ℝ) : ℝ := -g (-x)

-- Theorem statement
theorem symmetric_shifted_arctan :
  h = fun x ↦ Real.arctan (x + 2) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_shifted_arctan_l497_49780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_good_parabolas_l497_49700

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A parabola represented by a quadratic function --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point is on or above a parabola --/
def isOnOrAbove (p : Point) (para : Parabola) : Prop :=
  p.y ≥ para.a * p.x^2 + para.b * p.x + para.c

/-- Checks if a parabola is "good" --/
def isGoodParabola (points : List Point) (para : Parabola) (p1 p2 : Point) : Prop :=
  p1 ∈ points ∧ p2 ∈ points ∧ 
  ∀ p ∈ points, p ≠ p1 → p ≠ p2 → ¬(isOnOrAbove p para)

/-- The main theorem --/
theorem max_good_parabolas (n : ℕ) (points : List Point) :
  (points.length = n) →
  (∀ p1 p2, p1 ∈ points → p2 ∈ points → p1 ≠ p2 → p1.x ≠ p2.x) →
  ∃ goodParabolas : List Parabola, 
    (∀ para ∈ goodParabolas, ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ isGoodParabola points para p1 p2) ∧
    goodParabolas.length = n - 1 ∧
    ∀ otherParabolas : List Parabola, 
      (∀ para ∈ otherParabolas, ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ isGoodParabola points para p1 p2) →
      otherParabolas.length ≤ n - 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_good_parabolas_l497_49700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_rate_l497_49748

/-- The rate of descent for a diver given the depth and time -/
noncomputable def descent_rate (depth : ℝ) (time : ℝ) : ℝ := depth / time

/-- Theorem: A diver descending 4000 feet in 50 minutes has a descent rate of 80 feet per minute -/
theorem diver_descent_rate :
  let depth : ℝ := 4000
  let time : ℝ := 50
  descent_rate depth time = 80 := by
  -- Unfold the definition of descent_rate
  unfold descent_rate
  -- Simplify the expression
  simp
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_rate_l497_49748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l497_49712

theorem matrix_N_satisfies_conditions : ∃ (N : Matrix (Fin 2) (Fin 2) ℚ),
  N.mulVec (![2, -1] : Fin 2 → ℚ) = ![5, -3] ∧
  N.mulVec (![1, 4] : Fin 2 → ℚ) = ![-2, 8] ∧
  N = !![2, -1; -4/9, 19/9] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l497_49712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_inequality_l497_49744

theorem cubic_root_inequality (x : ℝ) :
  x^(1/3) + 1 / ((x^(1/3))^2 + 4) ≤ 0 ↔ x < -0.379 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_inequality_l497_49744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_twelve_l497_49767

/-- An arithmetic sequence with its first term and sum of first three terms -/
structure ArithmeticSequence where
  a₁ : ℚ
  s₃ : ℚ

/-- The n-th term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a₁ + (n - 1) * ((seq.s₃ - 3 * seq.a₁) / 3)

/-- Theorem stating that for the given arithmetic sequence, the 6th term is 12 -/
theorem sixth_term_is_twelve (seq : ArithmeticSequence) 
  (h1 : seq.a₁ = 2) 
  (h2 : seq.s₃ = 12) : 
  nthTerm seq 6 = 12 := by
  sorry

#eval nthTerm ⟨2, 12⟩ 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_twelve_l497_49767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quantity_count_l497_49743

theorem quantity_count (total_avg : ℝ) (subset1_count : ℕ) (subset1_avg : ℝ) 
                       (subset2_count : ℕ) (subset2_avg : ℝ) 
                       (h1 : total_avg = 8)
                       (h2 : subset1_count = 4)
                       (h3 : subset1_avg = 5)
                       (h4 : subset2_count = 2)
                       (h5 : subset2_avg = 14) :
  (subset1_count + subset2_count : ℕ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quantity_count_l497_49743
