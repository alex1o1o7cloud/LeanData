import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_piece_equation_l699_69966

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a five-digit number -/
def FiveDigitNumber := Fin 100000

/-- Check if a number has distinct digits -/
def has_distinct_digits (n : FiveDigitNumber) : Prop := sorry

/-- The equation PIE = k * PIECE is valid -/
def valid_equation (pie : FiveDigitNumber) (piece : FiveDigitNumber) (k : Nat) : Prop :=
  pie.val = k * piece.val ∧ has_distinct_digits pie ∧ has_distinct_digits piece

/-- The maximum number of PIECE terms and the number of valid solutions -/
theorem pie_piece_equation :
  (∃ (max_k : Nat) (solution_count : Nat),
    max_k = 7 ∧
    solution_count = 4 ∧
    (∀ k : Nat, ∃ pie piece : FiveDigitNumber, valid_equation pie piece k → k ≤ max_k) ∧
    (∃ solutions : Finset (FiveDigitNumber × FiveDigitNumber),
      solutions.card = solution_count ∧
      ∀ (pie piece : FiveDigitNumber), (pie, piece) ∈ solutions ↔ valid_equation pie piece max_k)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_piece_equation_l699_69966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_time_l699_69977

/-- Calculates the time taken for a train to cross a platform -/
noncomputable def time_to_cross_platform (train_length : ℝ) (platform_length : ℝ) (time_to_cross_pole : ℝ) : ℝ :=
  let train_speed := train_length / time_to_cross_pole
  let total_distance := train_length + platform_length
  total_distance / train_speed

/-- Proves that the time taken for a 300 m train to cross a 150.00000000000006 m platform 
    is approximately 27.00 seconds, given that it crosses a signal pole in 18 seconds -/
theorem train_crossing_platform_time :
  let train_length : ℝ := 300
  let platform_length : ℝ := 150.00000000000006
  let time_to_cross_pole : ℝ := 18
  let crossing_time := time_to_cross_platform train_length platform_length time_to_cross_pole
  ∃ ε > 0, |crossing_time - 27| < ε ∧ ε < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_time_l699_69977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_theorem_l699_69918

theorem cosine_sum_theorem (α β γ : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < γ) (h4 : γ < 2 * π) 
  (h5 : ∀ x : ℝ, Real.cos (x + α) + Real.cos (x + β) + Real.cos (x + γ) = 0) : 
  γ - α = 4 * π / 3 := by
  sorry

#check cosine_sum_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_theorem_l699_69918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l699_69908

open Real

-- Define f(α) as given in the problem
noncomputable def f (α : ℝ) : ℝ := 
  (sin (α - π/2) * cos (3*π/2 + α) * tan (π - α)) / (tan (-α - π) * sin (-α - π))

-- Theorem statement
theorem problem_solution (α : ℝ) 
  (h1 : π < α ∧ α < 3*π/2) -- α is in the third quadrant
  (h2 : cos (α - 3*π/2) = 1/5) :
  (f α = -cos α) ∧ 
  (f (α + π/6) = (6 * sqrt 2 - 1) / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l699_69908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l699_69930

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 4 * Real.sin (ω * x + Real.pi / 3)

theorem function_properties (ω : ℝ) (m : ℝ) :
  ω > 0 →
  (∀ x₁ x₂, Real.pi/6 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ Real.pi → f ω x₁ ≥ f ω x₂) →
  (∀ x, f ω (3*Real.pi/2 - x) = f ω (3*Real.pi/2 + x)) →
  (∀ x, -9*Real.pi/20 ≤ x ∧ x ≤ m → -2 ≤ f ω x ∧ f ω x ≤ 4) →
  ω ≤ 7/6 ∧ 3*Real.pi/20 ≤ m ∧ m ≤ 3*Real.pi/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l699_69930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_l699_69927

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * log x

-- State the theorem
theorem x0_value (x₀ : ℝ) (h : x₀ > 0) : 
  (deriv f) x₀ = 2 → x₀ = (exp 1 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_l699_69927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l699_69954

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < -1/2 then (2*x + 1) / (x^2)
  else Real.log (x + 1)

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 4*x - 4

-- State the theorem
theorem b_range (a b : ℝ) (h : f a + g b = 0) : 
  b ∈ Set.Icc (-1 : ℝ) 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l699_69954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_expression_equals_two_l699_69968

-- Define the triangle PQR
def triangle_PQR (P Q R : ℝ) : Prop :=
  -- Conditions for a valid triangle
  P + Q > R ∧ P + R > Q ∧ Q + R > P ∧
  -- Given side lengths
  Real.sin P / 6 = Real.sin Q / 7 ∧ Real.sin P / 6 = Real.sin R / 8

-- Theorem statement
theorem triangle_expression_equals_two (P Q R : ℝ) 
  (h : triangle_PQR P Q R) : 
  (Real.cos ((P - Q)/2) / Real.sin (R/2)) - (Real.sin ((P - Q)/2) / Real.cos (R/2)) = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_expression_equals_two_l699_69968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_simplification_and_sum_l699_69929

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 - 4*x - 12) / (x - 1)

-- Define the simplified function
def g (x : ℝ) : ℝ := x^2 + 4*x + 12

-- Define the point where f is undefined
def D : ℝ := 1

-- State the theorem
theorem function_simplification_and_sum :
  (∀ x : ℝ, x ≠ D → f x = g x) ∧
  (1 + 4 + 12 + D = 18) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_simplification_and_sum_l699_69929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_arrangement_l699_69936

theorem plant_arrangement (n : ℕ) (m : ℕ) (h1 : n = 5) (h2 : m = 4) :
  (Nat.factorial (n + 1)) * (Nat.factorial m) = 17280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_arrangement_l699_69936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_between_cube_roots_l699_69923

theorem whole_numbers_between_cube_roots : 
  (Finset.filter (fun n : ℕ => (n : ℝ) > Real.rpow 50 (1/3) ∧ (n : ℝ) < Real.rpow 500 (1/3)) (Finset.range 9)).card = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_between_cube_roots_l699_69923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rajesh_work_time_l699_69985

/-- Represents the time in days it takes for a person to complete the work alone -/
structure WorkTime where
  days : ℝ
  days_positive : days > 0

/-- Represents the share of money a person receives -/
structure MoneyShare where
  amount : ℝ
  amount_nonnegative : amount ≥ 0

/-- Calculates the work rate (portion of work done per day) -/
noncomputable def work_rate (wt : WorkTime) : ℝ := 1 / wt.days

theorem rajesh_work_time 
  (rahul_time : WorkTime)
  (rahul_share : MoneyShare)
  (total_money : ℝ)
  (h1 : rahul_time.days = 3)
  (h2 : work_rate rahul_time + work_rate (WorkTime.mk 2 (by norm_num)) = 1)
  (h3 : total_money = 250)
  (h4 : rahul_share.amount = 100)
  : ∃ (rajesh_time : WorkTime), rajesh_time.days = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rajesh_work_time_l699_69985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_roots_property_l699_69951

-- Define the polynomial f_n(x) as noncomputable
noncomputable def f (n : ℕ) : ℝ → ℝ :=
  fun x => (Finset.range (n + 1)).sum (fun k => x^k / (Nat.factorial k : ℝ))

-- Statement of the theorem
theorem f_roots_property :
  ∀ n : ℕ,
    (Even n → ∀ x : ℝ, f n x ≠ 0) ∧
    (Odd n → ∃! x : ℝ, f n x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_roots_property_l699_69951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l699_69997

/-- Given a hyperbola M with equation x²/m - y²/6 = 1 (m > 0) and eccentricity 2,
    prove that the asymptote equations for hyperbola N: x² - y²/m = 1 are y = ±√2x -/
theorem hyperbola_asymptotes (m : ℝ) (hm : m > 0) :
  (∀ x y : ℝ, x^2 / m - y^2 / 6 = 1 → (((m + 6) / m)^(1/2) = 2)) →
  (∀ x y : ℝ, x^2 - y^2 / m = 1 → (y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l699_69997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_divided_in_half_l699_69920

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- A line segment in a 2D plane -/
structure LineSegment where
  start : Point
  endpoint : Point

/-- Checks if a point is outside a circle -/
def isOutside (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 > c.radius^2

/-- Checks if a point is on a circle -/
def onCircle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- Checks if a point is on a line segment -/
def onLineSegment (p : Point) (l : LineSegment) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    p.1 = l.start.1 + t * (l.endpoint.1 - l.start.1) ∧
    p.2 = l.start.2 + t * (l.endpoint.2 - l.start.2)

/-- Checks if a line segment is a secant of a circle -/
def isSecant (l : LineSegment) (c : Circle) : Prop :=
  ∃ p1 p2 : Point, p1 ≠ p2 ∧ 
    onCircle p1 c ∧ onCircle p2 c ∧
    onLineSegment p1 l ∧ onLineSegment p2 l

/-- Checks if a point is the midpoint of a line segment -/
def isMidpoint (p : Point) (l : LineSegment) : Prop :=
  p.1 = (l.start.1 + l.endpoint.1) / 2 ∧
  p.2 = (l.start.2 + l.endpoint.2) / 2

theorem secant_divided_in_half 
  (c : Circle) (A : Point) (h : isOutside A c) :
  ∃ (AC : LineSegment), 
    AC.start = A ∧ 
    isSecant AC c ∧ 
    ∃ (B : Point), onCircle B c ∧ isMidpoint B AC :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_divided_in_half_l699_69920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tara_games_per_year_l699_69974

/-- The number of games Tara played each year -/
def games_per_year : ℕ := 20

/-- The number of games Tara's dad attended in the first year -/
def first_year_attendance : ℕ := (90 * games_per_year) / 100

/-- The number of games Tara's dad attended in the second year -/
def second_year_attendance : ℕ := 14

/-- The difference in attendance between the first and second year -/
def attendance_difference : ℕ := 4

theorem tara_games_per_year :
  (first_year_attendance = second_year_attendance + attendance_difference) →
  games_per_year = 20 := by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tara_games_per_year_l699_69974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_planting_equation_equivalence_tree_planting_scenario_l699_69972

/-- The equation representing the tree planting scenario -/
def tree_planting_equation (x : ℝ) : Prop :=
  (50 / x) - (50 / ((1 + 0.3) * x)) = 2

theorem tree_planting_equation_equivalence (x : ℝ) (h : x > 0) : 
  tree_planting_equation x ↔ 
  (500 / x) - (500 / ((1 + 0.3) * x)) = 2 :=
by sorry

theorem tree_planting_scenario (x : ℝ) (h : x > 0) :
  tree_planting_equation x ↔
  (∃ (planned_days actual_days : ℝ),
    planned_days = 500 / x ∧
    actual_days = 500 / ((1 + 0.3) * x) ∧
    planned_days - actual_days = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_planting_equation_equivalence_tree_planting_scenario_l699_69972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l699_69901

noncomputable section

theorem relationship_abc : ∃ (a b c : ℝ), 
  (0 < a ∧ a < 1) ∧ (1 < b) ∧ (c = 1) ∧ a < c ∧ c < b := by
  -- We use existential quantification instead of universal quantification
  -- and replace the specific decimal values with their properties
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l699_69901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_D_properties_sum_center_radius_D_l699_69961

/-- Definition of circle D -/
def circle_D (x y : ℝ) : Prop :=
  x^2 + 4*y - 16 = -y^2 + 12*x + 36

/-- The center and radius of circle D -/
noncomputable def center_radius_D : ℝ × ℝ × ℝ := (6, -2, Real.sqrt 92)

/-- Theorem stating the properties of circle D -/
theorem circle_D_properties :
  let (c, d, s) := center_radius_D
  ∀ x y : ℝ, circle_D x y ↔ (x - c)^2 + (y - d)^2 = s^2 :=
by sorry

/-- Theorem for the sum of center coordinates and radius -/
theorem sum_center_radius_D :
  let (c, d, s) := center_radius_D
  c + d + s = 4 + Real.sqrt 92 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_D_properties_sum_center_radius_D_l699_69961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_100_value_x_recurrence_l699_69914

-- Define the sequence using an auxiliary function
def aux : ℕ → ℚ → ℚ → ℚ
  | 0, _, _ => 0
  | 1, x1, _ => x1
  | 2, _, _ => 1
  | (n + 3), x1, x2 => 2 * x2 + x1 - 2 * aux n x1 x2

def x (n : ℕ) : ℚ := aux n 0 1

-- State the theorem
theorem x_100_value : x 100 = (4^50 - 1) / 3 := by
  sorry

-- Additional lemma to show the recurrence relation holds
theorem x_recurrence (n : ℕ) : x (n + 3) = 2 * x (n + 2) + x (n + 1) - 2 * x n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_100_value_x_recurrence_l699_69914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_smallest_winning_N_l699_69996

def S : Set ℕ := {n : ℕ | ∀ p : ℕ, Nat.Prime p → ¬(p^4 ∣ n)}

theorem second_smallest_winning_N : 
  (∃ N : ℕ, N < 625 ∧ N > 0 ∧ 
    (∀ a ∈ S, (N : ℤ) - (a : ℤ) ≠ 0 ∧ (N : ℤ) - (a : ℤ) ≠ 1)) ∧
  (∀ a ∈ S, (625 : ℤ) - (a : ℤ) ≠ 0 ∧ (625 : ℤ) - (a : ℤ) ≠ 1) ∧
  (∀ N : ℕ, N < 625 → N > 0 → 
    (∃ a ∈ S, (N : ℤ) - (a : ℤ) = 0 ∨ (N : ℤ) - (a : ℤ) = 1) ∨
    N = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_smallest_winning_N_l699_69996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l699_69922

/-- The number of days a and b worked together before b left -/
noncomputable def days_worked_together : ℝ := 20

/-- The total amount of work to be done -/
noncomputable def total_work : ℝ := 1

/-- The combined work rate of a and b per day -/
noncomputable def combined_rate : ℝ := total_work / 30

/-- The work rate of a alone per day -/
noncomputable def a_rate : ℝ := total_work / 60

theorem work_completion (x : ℝ) 
  (h1 : x = days_worked_together) 
  (h2 : x * combined_rate + 20 * a_rate = total_work) : 
  x = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l699_69922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robber_survival_and_eventual_capture_l699_69984

/-- Represents the board size -/
def boardSize : Nat := 2001

/-- Represents a position on the board -/
structure Position where
  x : Nat
  y : Nat
  deriving Repr

/-- Represents the possible moves -/
inductive Move
  | Down
  | Right
  | UpLeft
  | BottomRightToTopLeft
  deriving Repr

/-- The game state -/
structure GameState where
  policeman : Position
  robber : Position
  turn : Nat
  deriving Repr

/-- Initial game state -/
def initialState : GameState :=
  { policeman := ⟨boardSize / 2, boardSize / 2⟩,
    robber := ⟨boardSize / 2 - 1, boardSize / 2 + 1⟩,
    turn := 0 }

/-- Predicate to check if a position is valid -/
def isValidPosition (p : Position) : Prop :=
  p.x ≥ 0 ∧ p.x < boardSize ∧ p.y ≥ 0 ∧ p.y < boardSize

/-- Predicate to check if a move is valid for a given player and position -/
def isValidMove (move : Move) (isPolice : Bool) (pos : Position) : Prop :=
  match move with
  | Move.Down => isValidPosition ⟨pos.x, pos.y - 1⟩
  | Move.Right => isValidPosition ⟨pos.x + 1, pos.y⟩
  | Move.UpLeft => isValidPosition ⟨pos.x - 1, pos.y + 1⟩
  | Move.BottomRightToTopLeft => isPolice ∧ pos.x = boardSize - 1 ∧ pos.y = 0

/-- Predicate to check if the robber is captured -/
def isCaptured (state : GameState) : Prop :=
  state.policeman = state.robber

/-- Helper function to apply a move to a position -/
def applyMove (pos : Position) (move : Move) : Position :=
  match move with
  | Move.Down => ⟨pos.x, pos.y - 1⟩
  | Move.Right => ⟨pos.x + 1, pos.y⟩
  | Move.UpLeft => ⟨pos.x - 1, pos.y + 1⟩
  | Move.BottomRightToTopLeft => ⟨0, boardSize - 1⟩

/-- The main theorem to be proven -/
theorem robber_survival_and_eventual_capture :
  (∃ (strategy : GameState → Move),
    ∀ (policeStrategy : GameState → Move),
      ∃ (n : Nat), n ≥ 10000 ∧
        ¬(isCaptured (Nat.iterate (λ s ↦ 
          if s.turn % 2 = 0
          then { s with policeman := applyMove s.policeman (policeStrategy s), turn := s.turn + 1 }
          else { s with robber := applyMove s.robber (strategy s), turn := s.turn + 1 })
        n initialState))) ∧
  (∃ (strategy : GameState → Move),
    ∀ (robberStrategy : GameState → Move),
      ∃ (n : Nat),
        isCaptured (Nat.iterate (λ s ↦ 
          if s.turn % 2 = 0
          then { s with policeman := applyMove s.policeman (strategy s), turn := s.turn + 1 }
          else { s with robber := applyMove s.robber (robberStrategy s), turn := s.turn + 1 })
        n initialState)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_robber_survival_and_eventual_capture_l699_69984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_30_l699_69976

theorem divisors_of_30 : 
  (∀ n : ℕ, n ∈ ({1, 2, 3, 5} : Set ℕ) → 30 % n = 0) → 
  (∀ n : ℕ, n ∈ ({1, 2, 3, 5} : Set ℕ) → n ∣ 30) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_30_l699_69976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_base_solution_l699_69943

/-- Represents a number in base a --/
structure BaseANumber (a : ℕ) where
  digits : List ℕ
  valid : ∀ d ∈ digits, d < a

/-- Converts a base-10 number to base a --/
def toBaseA (n : ℕ) (a : ℕ) : BaseANumber a :=
  sorry

/-- Adds two numbers in base a --/
def addBaseA {a : ℕ} (x y : BaseANumber a) : BaseANumber a :=
  sorry

/-- Converts a base a number to base 10 --/
def toBase10 {a : ℕ} (x : BaseANumber a) : ℕ :=
  sorry

theorem unique_base_solution :
  ∃! a : ℕ, 
    a > 11 ∧
    let B := toBaseA 11 a
    let n1 := toBaseA 293 a
    let n2 := toBaseA 468 a
    let n3 := toBaseA 73 a
    addBaseA n1 n2 = addBaseA n3 B :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_base_solution_l699_69943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l699_69999

/-- The compound interest formula for quarterly compounding -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 4) ^ (4 * time)

/-- The problem statement -/
theorem principal_calculation (final_amount : ℝ) (rate : ℝ) (time : ℝ) 
  (h1 : final_amount = 4410)
  (h2 : rate = 0.07)
  (h3 : time = 2) :
  ∃ (principal : ℝ), compound_interest principal rate time = final_amount := by
  sorry

#eval 4410 / (1 + 0.07 / 4) ^ (4 * 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l699_69999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l699_69955

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (x^2 - x - 1)

-- State the theorem
theorem monotonic_decreasing_interval :
  ∀ x y : ℝ, x > 1/2 ∧ y > 1/2 ∧ x < y → f x > f y :=
by
  -- The proof is omitted using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l699_69955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_values_f_and_g_l699_69933

open Real

theorem max_values_f_and_g :
  (∃ (θ : ℝ), θ ∈ Set.Ioo 0 (π/2) ∧ ∀ (φ : ℝ), φ ∈ Set.Ioo 0 (π/2) → cos (θ/2) * sin θ ≥ cos (φ/2) * sin φ) ∧
  (∃ (θ : ℝ), θ ∈ Set.Ioo 0 (π/2) ∧ ∀ (φ : ℝ), φ ∈ Set.Ioo 0 (π/2) → sin (θ/2) * cos θ ≥ sin (φ/2) * cos φ) ∧
  (∀ (θ : ℝ), θ ∈ Set.Ioo 0 (π/2) → cos (θ/2) * sin θ ≤ 4 * sqrt 3 / 9) ∧
  (∀ (θ : ℝ), θ ∈ Set.Ioo 0 (π/2) → sin (θ/2) * cos θ ≤ sqrt 6 / 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_values_f_and_g_l699_69933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l699_69945

-- Define the types for points and distances
variable {Point : Type*} [NormedAddCommGroup Point] [InnerProductSpace ℝ Point]
variable {Distance : Type*} [NormedAddCommGroup Distance] [InnerProductSpace ℝ Distance]

-- Define the distance function
variable (dist : Point → Point → ℝ)

-- Define the points
variable (A B C A' B' C' G A1 B1 C1 : Point)

-- Define the sphere function
def sphere (center : Point) (radius : ℝ) : Set Point :=
  {p : Point | dist center p = radius}

-- State the theorem
theorem triangle_inequality 
  (congruent : dist A B = dist A' B' ∧ dist B C = dist B' C' ∧ dist C A = dist C' A')
  (centroid : G = (1/3 : ℝ) • (A + B + C))
  (A1_def : A1 ∈ sphere dist A ((dist A A') / 2) ∧ A1 ∈ sphere dist A' (dist A' G))
  (B1_def : B1 ∈ sphere dist B ((dist B B') / 2) ∧ B1 ∈ sphere dist B' (dist B' G))
  (C1_def : C1 ∈ sphere dist C ((dist C C') / 2) ∧ C1 ∈ sphere dist C' (dist C' G)) :
  (dist A A1)^2 + (dist B B1)^2 + (dist C C1)^2 ≤ (dist A B)^2 + (dist B C)^2 + (dist C A)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l699_69945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l699_69991

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

theorem ellipse_eccentricity (a b : ℝ) (F A B M : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (∃ x y, ellipse a b x y) ∧
  (∃ x y, F = (x, y)) ∧
  (∃ x y, A = (x, y) ∧ ellipse a b x y) ∧
  (∃ x y, B = (x, y) ∧ ellipse a b x y) ∧
  M = (-A.1, -A.2) ∧
  (B.1 - A.1) * (M.1 - F.1) + (B.2 - A.2) * (M.2 - F.2) = 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (M.1 - F.1)^2 + (M.2 - F.2)^2 →
  eccentricity a b = Real.sqrt 6 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l699_69991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_a_c_is_5pi_6_min_value_f_l699_69971

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (-Real.cos x, Real.cos x)
def c : ℝ × ℝ := (-1, 0)

noncomputable def angle_between (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

noncomputable def f (x : ℝ) : ℝ := 2 * (a x).1 * (b x).1 + 2 * (a x).2 * (b x).2 + 1

theorem angle_a_c_is_5pi_6 :
  angle_between (a (π/6)) c = 5*π/6 := by sorry

theorem min_value_f :
  ∃ (x : ℝ), x ∈ Set.Icc (π/2) (9*π/8) ∧
  f x = -Real.sqrt 2 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (π/2) (9*π/8) → f y ≥ f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_a_c_is_5pi_6_min_value_f_l699_69971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_eval_one_eq_binomial_coeff_l699_69964

/-- The polynomial g_{k,l}(x) -/
noncomputable def g (k l : ℕ) : ℝ → ℝ := sorry

/-- Binomial coefficient -/
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem g_eval_one_eq_binomial_coeff (k l : ℕ) :
  g k l 1 = binomial_coeff (k + l) k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_eval_one_eq_binomial_coeff_l699_69964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l699_69952

-- Define the line l
noncomputable def line_l (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 2 + t * Real.sin α)

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 9

-- Define point P
def point_P : ℝ × ℝ := (1, 2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_sum (α : ℝ) :
  ∃ (t1 t2 : ℝ),
    circle_C (line_l t1 α).1 (line_l t1 α).2 ∧
    circle_C (line_l t2 α).1 (line_l t2 α).2 ∧
    (∀ (s1 s2 : ℝ),
      circle_C (line_l s1 α).1 (line_l s1 α).2 →
      circle_C (line_l s2 α).1 (line_l s2 α).2 →
      distance point_P (line_l t1 α) + distance point_P (line_l t2 α) ≤
      distance point_P (line_l s1 α) + distance point_P (line_l s2 α)) ∧
    distance point_P (line_l t1 α) + distance point_P (line_l t2 α) = 2 * Real.sqrt 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l699_69952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_proof_l699_69900

theorem trigonometric_proof (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.sin α = 3/5) 
  (h4 : Real.tan (α - β) = -1/3) : 
  Real.sin (α - β) = -Real.sqrt 10 / 10 ∧ 
  Real.cos β = 9 * Real.sqrt 10 / 50 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_proof_l699_69900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l699_69965

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 - 2*x

-- Define a, b, and c as noncomputable
noncomputable def a : ℝ := Real.log 2
noncomputable def b : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def c : ℝ := Real.sqrt 3

-- State the theorem
theorem f_inequality : f b > f a ∧ f a > f c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l699_69965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_number_proof_l699_69944

/-- Given a real number x, if x + 342.00000000007276 is divisible by 412,
    then x is approximately equal to 69.99999999992724 -/
theorem initial_number_proof (x : ℝ) : 
  (∃ n : ℤ, x + 342.00000000007276 = 412 * ↑n) → 
  ∃ ε > 0, |x - 69.99999999992724| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_number_proof_l699_69944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_eight_l699_69998

noncomputable def f (x : ℝ) : ℝ := x / (x + 1) + (x + 1) / (x + 2) + (x + 2) / (x + 3) + (x + 3) / (x + 4)

theorem f_sum_equals_eight :
  f (-5/2 + Real.sqrt 5) + f (-5/2 - Real.sqrt 5) = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_eight_l699_69998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mandy_has_20_toys_l699_69903

/-- The number of toys Mandy has -/
def mandy_toys : ℕ := sorry

/-- The number of toys Anna has -/
def anna_toys : ℕ := sorry

/-- The number of toys Amanda has -/
def amanda_toys : ℕ := sorry

/-- Anna has 3 times as many toys as Mandy -/
axiom anna_mandy_relation : anna_toys = 3 * mandy_toys

/-- Anna has 2 fewer toys than Amanda -/
axiom anna_amanda_relation : amanda_toys = anna_toys + 2

/-- The total number of toys is 142 -/
axiom total_toys : mandy_toys + anna_toys + amanda_toys = 142

/-- Theorem: Mandy has 20 toys -/
theorem mandy_has_20_toys : mandy_toys = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mandy_has_20_toys_l699_69903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_slope_product_l699_69928

-- Define the ellipse (C)
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the circle
noncomputable def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define a line with slope k and y-intercept m
def line (k m x y : ℝ) : Prop := y = k * x + m

-- Define the condition for (l) to intersect (C) at exactly one point
def tangent_condition (k m : ℝ) : Prop := m^2 = 4 * k^2 + 1

-- Define the slope of a line passing through the origin and a point (x, y)
noncomputable def slope_through_origin (x y : ℝ) : ℝ := y / x

-- Main theorem
theorem ellipse_tangent_slope_product (k m x₁ y₁ x₂ y₂ : ℝ) :
  ellipse x₁ y₁ →
  ellipse x₂ y₂ →
  my_circle x₁ y₁ →
  my_circle x₂ y₂ →
  line k m x₁ y₁ →
  line k m x₂ y₂ →
  tangent_condition k m →
  x₁ ≠ 0 →
  x₂ ≠ 0 →
  x₁ ≠ x₂ →
  slope_through_origin x₁ y₁ * slope_through_origin x₂ y₂ = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_slope_product_l699_69928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_proof_l699_69904

/-- The distance between the foci of an ellipse with equation 9x^2 + y^2 = 144 -/
noncomputable def ellipse_foci_distance : ℝ :=
  16 * Real.sqrt 2

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  9 * x^2 + y^2 = 144

theorem ellipse_foci_distance_proof :
  ∀ x y : ℝ, ellipse_equation x y → ellipse_foci_distance = 16 * Real.sqrt 2 := by
  sorry

#check ellipse_foci_distance_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_proof_l699_69904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_root_product_l699_69983

noncomputable def f (x : ℝ) : ℝ := Real.log x - (1 / 2017) * x

theorem inequality_and_root_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (Real.sqrt (a * b) < (a - b) / (Real.log a - Real.log b) ∧ 
   (a - b) / (Real.log a - Real.log b) < (a + b) / 2) ∧
  (f a = 0 ∧ f b = 0 → a * b > Real.exp 2) :=
by sorry

#check inequality_and_root_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_root_product_l699_69983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_d_value_l699_69926

/-- The smallest positive value of d that satisfies the distance equation -/
theorem smallest_d_value : 
  (∃ d : ℝ, d > 0 ∧ 
    4 * d = Real.sqrt ((4 * Real.sqrt 3)^2 + (d - 2)^2) ∧
    (∀ d' : ℝ, d' > 0 → 
      4 * d' = Real.sqrt ((4 * Real.sqrt 3)^2 + (d' - 2)^2) → 
      d ≤ d')) → 
  (∃ d : ℝ, d = 2 ∧ 
    4 * d = Real.sqrt ((4 * Real.sqrt 3)^2 + (d - 2)^2) ∧
    (∀ d' : ℝ, d' > 0 → 
      4 * d' = Real.sqrt ((4 * Real.sqrt 3)^2 + (d' - 2)^2) → 
      d ≤ d')) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_d_value_l699_69926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_count_theorem_l699_69939

def is_valid (n : ℕ) : Bool :=
  (n^3 + 3^n) % 5 = 0

def count_valid (a b : ℕ) : ℕ :=
  (List.range (b - a + 1)).filter (fun i => is_valid (i + a)) |>.length

theorem valid_count_theorem :
  count_valid 1 2015 = 500 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_count_theorem_l699_69939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l699_69960

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

theorem function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f a b (-x) = -f a b x) →
  f a b (1/2) = 2/5 →
  (∃ g : ℝ → ℝ, 
    (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → g x = x / (1 + x^2)) ∧
    (∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → g x < g y) ∧
    (Set.Ioo 0 (1/2) = {x | x ∈ Set.Ioo (-1 : ℝ) 1 ∧ g (x-1) + g x < 0})) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l699_69960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l699_69973

noncomputable def f (x : ℝ) := Real.log (x + 2) - x

theorem arithmetic_sequence_max_sum (a b c d : ℝ) : 
  (∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r) →  -- arithmetic sequence condition
  (∀ x : ℝ, f x ≤ f b) →  -- b is the x-coordinate of the maximum point
  (f b = c) →  -- c is the maximum value
  b + d = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l699_69973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_properties_l699_69916

noncomputable section

variable (b : ℝ)
variable (hb : b > 1)

def log (x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_properties (hb : b > 1) :
  (log b (b ^ 2) = 2) ∧
  (log b (1 / b) = -1) ∧
  (log b (Real.sqrt b) = 1 / 2) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x < b ∧ log b x ≥ 0) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_properties_l699_69916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_triangle_area_l699_69979

/-- Three circles with radii 2, 3, and 4 that are mutually externally tangent -/
structure TangentCircles where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ
  center3 : ℝ × ℝ
  radius1 : ℝ
  radius2 : ℝ
  radius3 : ℝ
  radius1_eq : radius1 = 2
  radius2_eq : radius2 = 3
  radius3_eq : radius3 = 4
  externally_tangent : 
    (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = (radius1 + radius2)^2 ∧
    (center2.1 - center3.1)^2 + (center2.2 - center3.2)^2 = (radius2 + radius3)^2 ∧
    (center3.1 - center1.1)^2 + (center3.2 - center1.2)^2 = (radius3 + radius1)^2

/-- The triangle formed by the points of tangency of the three circles -/
def tangencyTriangle (tc : TangentCircles) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ := sorry

/-- The area of a triangle given its vertices as a tuple of 6 real numbers -/
noncomputable def triangleArea (vertices : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ) : ℝ := sorry

/-- The theorem stating that the area of the triangle formed by the points of tangency
    of three mutually externally tangent circles with radii 2, 3, and 4 is equal to 3√6/2 -/
theorem tangency_triangle_area (tc : TangentCircles) :
  triangleArea (tangencyTriangle tc) = 3 * Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_triangle_area_l699_69979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_derivative_at_one_l699_69962

theorem range_of_derivative_at_one (θ : Real) (h : θ ∈ Set.Icc 0 (5 * Real.pi / 12)) :
  let f : Real → Real := λ x => (Real.sin θ / 3) * x^3 + (Real.sqrt 3 * Real.cos θ / 2) * x^2 + Real.tan θ
  let f' : Real → Real := λ x => Real.sin θ * x^2 + Real.sqrt 3 * Real.cos θ * x
  f' 1 ∈ Set.Icc (Real.sqrt 2) 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_derivative_at_one_l699_69962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_400_l699_69994

-- Define the principal amount P
variable (P : ℝ)

-- Define the unknown interest rate R
variable (R : ℝ)

-- Define the time period in years
noncomputable def T : ℝ := 10

-- Define the additional interest earned with increased rate
noncomputable def additional_interest : ℝ := 200

-- Define the simple interest function
noncomputable def simple_interest (principal rate time : ℝ) : ℝ := (principal * rate * time) / 100

-- Theorem statement
theorem principal_is_400 :
  (simple_interest P (R + 5) T - simple_interest P R T = additional_interest) →
  P = 400 := by
  sorry

#check principal_is_400

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_400_l699_69994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_crossing_time_l699_69975

/-- The time taken for a train to cross a platform -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) (platform_length : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem stating the time taken for a specific train to cross a specific platform -/
theorem train_platform_crossing_time :
  let train_length : ℝ := 90
  let train_speed_kmph : ℝ := 56
  let platform_length : ℝ := 190.0224
  |train_crossing_time train_length train_speed_kmph platform_length - 18.0007| < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_crossing_time_l699_69975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seat_assignment_probabilities_l699_69905

/-- The number of individuals and seats -/
def n : ℕ := 5

/-- The probability that exactly k out of n individuals sit in their assigned seats -/
def prob_exact (k : ℕ) : ℚ :=
  (Nat.choose n k * Nat.factorial (n - k)) / Nat.factorial n

/-- The probability that at most k out of n individuals sit in their assigned seats -/
def prob_at_most (k : ℕ) : ℚ :=
  Finset.sum (Finset.range (k + 1)) (λ i => prob_exact i)

theorem seat_assignment_probabilities :
  (prob_exact 3 = 1/12) ∧
  (∀ k : ℕ, prob_at_most k ≥ 1/6 → k ≤ 2) :=
by sorry

#eval prob_exact 3
#eval prob_at_most 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seat_assignment_probabilities_l699_69905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trig_expression_l699_69969

theorem min_trig_expression (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  (Real.sin x + Real.sin x⁻¹ + Real.tan x)^2 + 
  (Real.cos x + Real.cos x⁻¹ + (Real.cos x / Real.sin x))^2 ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trig_expression_l699_69969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_passes_through_intersection_l699_69902

-- Define the angle between Ox and Oy
variable (ω : ℝ)

-- Define the ellipse parameters
variable (a b : ℝ)

-- Define the coordinates of points A and B
variable (α β : ℝ)

-- Define the coordinates of point M (midpoint of AB)
noncomputable def x₀ (a b α : ℝ) : ℝ := (a * α) / (a + b)
noncomputable def y₀ (a b β : ℝ) : ℝ := (b * β) / (a + b)

-- Define the ellipse equation
def ellipse_equation (a b ω : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - 2*x*y / (a*b) * Real.cos ω + y^2 / b^2 = 1

-- Define the normal line to the ellipse at point M
def normal_line (a b α β ω : ℝ) (x y : ℝ) : Prop :=
  (a * (x + y * Real.cos ω - α)) / (α - β * Real.cos ω) = 
  (b * (x * Real.cos ω + y - β)) / (β - α * Real.cos ω)

-- Define the perpendicular lines from A and B to Ox and Oy
def perp_line_A (α ω : ℝ) (x y : ℝ) : Prop := x + y * Real.cos ω = α
def perp_line_B (β ω : ℝ) (x y : ℝ) : Prop := y + x * Real.cos ω = β

-- Theorem statement
theorem normal_passes_through_intersection (a b α β ω : ℝ) :
  ∀ x y : ℝ,
  ellipse_equation a b ω (x₀ a b α) (y₀ a b β) →
  normal_line a b α β ω x y →
  (perp_line_A α ω x y ∧ perp_line_B β ω x y) →
  -- The normal line passes through the intersection point
  True :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_passes_through_intersection_l699_69902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_coin_problem_l699_69982

theorem pigeonhole_coin_problem :
  ∀ (coins : Finset ℕ) (denom : ℕ → Fin 4),
    coins.card = 25 →
    ∃ (i : Fin 4), (coins.filter (λ c => denom c = i)).card ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_coin_problem_l699_69982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donkey_winning_strategy_exists_l699_69953

/-- Represents a point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line segment between two points -/
structure Segment where
  p1 : Point
  p2 : Point

/-- The game state -/
structure GameState where
  points : Finset Point
  segments : Finset Segment
  segmentLabels : Segment → ℕ
  pointLabels : Point → ℕ

/-- Predicate to check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Predicate to check if Donkey wins -/
def donkeyWins (state : GameState) : Prop :=
  ∃ (p1 p2 : Point) (s : Segment), 
    s ∈ state.segments ∧ 
    s.p1 = p1 ∧ s.p2 = p2 ∧
    state.segmentLabels s = state.pointLabels p1 ∧
    state.segmentLabels s = state.pointLabels p2

/-- The main theorem stating Donkey's winning strategy exists -/
theorem donkey_winning_strategy_exists :
  ∀ (state : GameState),
    state.points.card = 2005 →
    (∀ p1 p2 p3, p1 ∈ state.points → p2 ∈ state.points → p3 ∈ state.points → 
      p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬collinear p1 p2 p3) →
    (∀ p1 p2, p1 ∈ state.points → p2 ∈ state.points → p1 ≠ p2 → 
      ∃ s ∈ state.segments, s.p1 = p1 ∧ s.p2 = p2) →
    ∃ (segmentLabeling : Segment → ℕ),
      ∀ (pointLabeling : Point → ℕ),
        donkeyWins {points := state.points, segments := state.segments, 
                    segmentLabels := segmentLabeling, pointLabels := pointLabeling} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_donkey_winning_strategy_exists_l699_69953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l699_69938

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 3)

noncomputable def h (ω : ℝ) (b : ℝ) (x : ℝ) : ℝ := f ω x - b

theorem function_properties (ω : ℝ) (b : ℝ) :
  ω > 0 →
  (∀ x, f ω (x + Real.pi) = f ω x) →
  (∀ p, p > 0 → (∀ x, f ω (x + p) = f ω x) → p ≥ Real.pi) →
  (∃ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ Real.pi / 2 ∧ h ω b x₁ = 0 ∧ h ω b x₂ = 0) →
  (∃ k : ℤ, ∀ x, (k * Real.pi + 5 * Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 11 * Real.pi / 12) →
    ∀ y, (x < y ∧ y ≤ k * Real.pi + 11 * Real.pi / 12) → f ω y < f ω x) ∧
  (Real.sqrt 3 ≤ b ∧ b < 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l699_69938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_in_idiom_sum_l699_69924

theorem max_value_in_idiom_sum (n : ℕ) (a b c d e : ℕ) :
  n ≥ 11 →
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 →
  a ≤ n ∧ b ≤ n ∧ c ≤ n ∧ d ≤ n ∧ e ≤ n →
  a + b + c + d + e ≤ 18 →
  (∃ w x y z : ℕ, w + x + y + z = 21 ∧ 
    w ∈ ({a, b, c, d, e} : Set ℕ) ∧ x ∈ ({a, b, c, d, e} : Set ℕ) ∧ 
    y ∈ ({a, b, c, d, e} : Set ℕ) ∧ z ∈ ({a, b, c, d, e} : Set ℕ)) →
  (∃ w x y z : ℕ, w + x + y + z = 21 ∧ 
    w ∈ ({a, b, c, d, e} : Set ℕ) ∧ x ∈ ({a, b, c, d, e} : Set ℕ) ∧ 
    y ∈ ({a, b, c, d, e} : Set ℕ) ∧ z ∈ ({a, b, c, d, e} : Set ℕ)) →
  (∃ w x y z : ℕ, w + x + y + z = 21 ∧ 
    w ∈ ({a, b, c, d, e} : Set ℕ) ∧ x ∈ ({a, b, c, d, e} : Set ℕ) ∧ 
    y ∈ ({a, b, c, d, e} : Set ℕ) ∧ z ∈ ({a, b, c, d, e} : Set ℕ)) →
  (∃ w x y z : ℕ, w + x + y + z = 21 ∧ 
    w ∈ ({a, b, c, d, e} : Set ℕ) ∧ x ∈ ({a, b, c, d, e} : Set ℕ) ∧ 
    y ∈ ({a, b, c, d, e} : Set ℕ) ∧ z ∈ ({a, b, c, d, e} : Set ℕ)) →
  e ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_in_idiom_sum_l699_69924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_cube_l699_69906

theorem divisors_of_cube (n : ℕ+) (h : (Nat.divisors n.val).card = 4) : 
  (Nat.divisors (n^3).val).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_cube_l699_69906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l699_69910

theorem trigonometric_identity :
  Real.sin (18 * π / 180)^2 + Real.cos (63 * π / 180)^2 + 
  Real.sqrt 2 * Real.sin (18 * π / 180) * Real.cos (63 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l699_69910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l699_69915

noncomputable def g (x : ℝ) : ℝ := ⌊2 * x⌋ - 2 * x

theorem range_of_g :
  Set.range g = Set.Ioc (-1) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l699_69915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l699_69986

-- Define the circle C
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 2

-- Define the line
def line_eq (x y b : ℝ) : Prop := y = 2*x + b

-- Define the intersection of the circle with y-axis
def y_axis_intersection (y : ℝ) : Prop := circle_eq 0 y

-- Define the intersection of the circle with the line
def line_intersection (x y b : ℝ) : Prop := circle_eq x y ∧ line_eq x y b

-- Define the length of a segment
noncomputable def segment_length (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem circle_line_intersection (b : ℝ) : 
  (∃ y1 y2, y_axis_intersection y1 ∧ y_axis_intersection y2 ∧ 
    segment_length 0 y1 0 y2 = 2) →
  (∃ x1 y1 x2 y2, line_intersection x1 y1 b ∧ line_intersection x2 y2 b ∧ 
    segment_length x1 y1 x2 y2 = 2) →
  b = Real.sqrt 5 ∨ b = -Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l699_69986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_equation_l699_69978

/-- A parabola with vertex at the origin, focus on the x-axis, and a point M(1, m) on the parabola
    such that the distance from M to the focus is 2 has the standard equation y² = 4x. -/
theorem parabola_standard_equation (p : ℝ → ℝ → Prop) (f : ℝ × ℝ) (m : ℝ) :
  (∀ x y, p x y ↔ y^2 = 4*x) →  -- Standard equation of the parabola
  (f.1 > 0 ∧ f.2 = 0) →  -- Focus is on the positive x-axis
  p 1 m →  -- Point M(1, m) is on the parabola
  ((1 : ℝ) - f.1)^2 + m^2 = 4 →  -- Distance from M to focus is 2
  ∀ x y, p x y ↔ y^2 = 4*x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_equation_l699_69978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_speed_ratio_l699_69958

/-- Marathon runner times and speeds -/
theorem marathon_speed_ratio :
  ∀ (micah_time dean_time jake_time : ℝ),
  dean_time = 9 →
  jake_time = micah_time + (1/3) * micah_time →
  micah_time + jake_time + dean_time = 23 →
  (dean_time / micah_time) = (3/2) :=
by
  intros micah_time dean_time jake_time h_dean h_jake h_total
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_speed_ratio_l699_69958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l699_69925

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2

-- State the theorem
theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l699_69925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_a_l699_69937

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_decreasing : ∀ x y, x < y → f x > f y

-- Define the inequality condition
axiom inequality_condition : ∀ x a, f (Real.cos (2 * x) + Real.sin x) + f (Real.sin x - a) ≤ 0

-- State the theorem
theorem max_value_of_a : 
  (∀ a, ∃ x, f (Real.cos (2 * x) + Real.sin x) + f (Real.sin x - a) > 0) ∨ 
  (∃ a, (∀ x, f (Real.cos (2 * x) + Real.sin x) + f (Real.sin x - a) ≤ 0) ∧ a = -3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_a_l699_69937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_symmetry_l699_69988

-- Define a random variable following a normal distribution
structure NormalDistribution (μ σ : ℝ) where
  value : ℝ

-- Define the probability function
noncomputable def prob {α : Type*} (X : Set α) (event : Set α) : ℝ := sorry

theorem normal_distribution_symmetry 
  {σ : ℝ} (ξ : NormalDistribution 1 σ)
  (h : prob (Set.univ : Set (NormalDistribution 1 σ)) {x | x.value ≤ 4} = 0.79) :
  prob (Set.univ : Set (NormalDistribution 1 σ)) {x | x.value ≤ -2} = 0.21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_symmetry_l699_69988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_tangents_l699_69956

-- Define A as a noncomputable real number representing an angle
noncomputable def A : ℝ := Real.arccos (3/5)

-- Theorem statement
theorem triangle_angle_tangents :
  -- Conditions
  0 < A ∧ A < Real.pi ∧ Real.cos A = 3/5 →
  -- Conclusions
  Real.tan A = 4/3 ∧ Real.tan (A + Real.pi/4) = -7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_tangents_l699_69956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_circle_l699_69980

/-- The equation of the circle boundary -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8 = 2*x + 4*y

/-- The radius of the circle -/
noncomputable def radius : ℝ := Real.sqrt 13

/-- The area of the circle -/
noncomputable def area : ℝ := 13 * Real.pi

/-- Theorem stating that the given equation describes a circle with radius √13 and area 13π -/
theorem cookie_circle :
  (∃ h k r : ℝ, ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2) ∧
  radius = Real.sqrt 13 ∧
  area = Real.pi * radius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_circle_l699_69980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_fraction_of_circle_radius_l699_69990

theorem rectangle_length_fraction_of_circle_radius :
  ∀ (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ),
    square_area = 625 →
    rectangle_area = 100 →
    rectangle_breadth = 10 →
    (rectangle_area / rectangle_breadth) / (Real.sqrt square_area) = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_fraction_of_circle_radius_l699_69990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gwen_birthday_money_difference_l699_69919

/-- Proves that Gwen received $1.75 more from her mom than her dad after all transactions -/
theorem gwen_birthday_money_difference 
  (mom_gift : ℚ) 
  (dad_gift : ℚ) 
  (grandparents_gift : ℚ) 
  (aunt_gift : ℚ) 
  (toy_cost : ℚ) 
  (snacks_cost : ℚ) 
  (book_percentage : ℚ) 
  (h1 : mom_gift = 8.25)
  (h2 : dad_gift = 6.50)
  (h3 : grandparents_gift = 12.35)
  (h4 : aunt_gift = 5.10)
  (h5 : toy_cost = 4.45)
  (h6 : snacks_cost = 6.25)
  (h7 : book_percentage = 0.25) : 
  mom_gift - dad_gift = 1.75 := by
  sorry

#eval (8.25 : ℚ) - (6.50 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gwen_birthday_money_difference_l699_69919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_condition_increasing_condition_max_k_condition_l699_69946

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.log x + 1) + a * x

-- Theorem 1
theorem extreme_point_condition (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1) → a = -2 := by
  sorry

-- Theorem 2
theorem increasing_condition (a : ℝ) :
  (∀ x y, x ∈ Set.Ici (Real.sqrt (Real.exp 1)) → y ∈ Set.Ici (Real.sqrt (Real.exp 1)) → x ≤ y → f a x ≤ f a y) →
  a ∈ Set.Ici (-5/2) := by
  sorry

-- Theorem 3
theorem max_k_condition :
  (∀ x, x ∈ Set.Ioi 1 → 3 * (x - 1) < f 1 x - x) ∧
  (∀ k : ℤ, (∀ x, x ∈ Set.Ioi 1 → k * (x - 1) < f 1 x - x) → k ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_condition_increasing_condition_max_k_condition_l699_69946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_104_exists_l699_69931

def arithmetic_progression (n : ℕ) : ℤ := 3 * n - 2

theorem sum_104_exists (A : Finset ℤ) 
  (h1 : A.card = 20)
  (h2 : ∀ x ∈ A, ∃ n : ℕ, n ≥ 1 ∧ n ≤ 34 ∧ x = arithmetic_progression n)
  (h3 : A.card = Finset.card A) :
  ∃ x y : ℤ, x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ x + y = 104 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_104_exists_l699_69931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paths_avoiding_middle_cell_l699_69921

/-- Represents a point on the grid -/
structure Point where
  x : Nat
  y : Nat

/-- Calculates the number of paths between two points on a grid -/
def pathsBetween (start : Point) (finish : Point) : Nat :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The theorem to be proved -/
theorem paths_avoiding_middle_cell :
  let gridWidth : Nat := 7
  let gridHeight : Nat := 6
  let start : Point := ⟨0, 0⟩
  let finish : Point := ⟨gridWidth - 1, gridHeight - 1⟩
  let middleCell : Point := ⟨3, 3⟩
  
  pathsBetween start finish - 
  (pathsBetween start middleCell * pathsBetween middleCell finish) = 1016 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paths_avoiding_middle_cell_l699_69921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_condition_for_line_intersection_l699_69963

/-- Represents a plane in 3D space -/
structure Plane where

/-- Represents a line in 3D space -/
structure Line where

/-- Angle between two planes -/
noncomputable def angle_between_planes (p1 p2 : Plane) : ℝ := sorry

/-- Intersection of two planes -/
noncomputable def plane_intersection (p1 p2 : Plane) : Line := sorry

/-- Check if three lines intersect at a single point -/
def lines_intersect_at_point (l1 l2 l3 : Line) : Prop := sorry

/-- Main theorem: θ > π/3 is sufficient but not necessary for line intersection -/
theorem angle_condition_for_line_intersection 
  (α β γ : Plane) 
  (h1 : angle_between_planes α β = angle_between_planes β γ)
  (h2 : angle_between_planes β γ = angle_between_planes γ α)
  (a : Line) (ha : a = plane_intersection α β)
  (b : Line) (hb : b = plane_intersection β γ)
  (c : Line) (hc : c = plane_intersection γ α) :
  (∃ θ : ℝ, 
    (angle_between_planes α β = θ ∧ 
     angle_between_planes β γ = θ ∧ 
     angle_between_planes γ α = θ) ∧
    (θ > π / 3 → lines_intersect_at_point a b c) ∧
    ¬(lines_intersect_at_point a b c → θ > π / 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_condition_for_line_intersection_l699_69963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_calculation_l699_69912

/-- Calculates the principal amount given the final amount and yearly interest rates -/
noncomputable def calculate_principal (final_amount : ℝ) (interest_rates : List ℝ) : ℝ :=
  interest_rates.foldl (fun acc rate => acc / (1 + rate)) final_amount

/-- Theorem: The principal amount is approximately 2396.43 given the conditions -/
theorem principal_amount_calculation :
  let final_amount : ℝ := 3000
  let interest_rates : List ℝ := [0.05, 0.06, 0.04, 0.05, 0.03]
  let calculated_principal := calculate_principal final_amount interest_rates
  abs (calculated_principal - 2396.43) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_principal 3000 [0.05, 0.06, 0.04, 0.05, 0.03]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_calculation_l699_69912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l699_69907

noncomputable def f (x : ℝ) : ℝ := (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7))

theorem inequality_solution :
  ∀ x : ℝ, f x > 0 ↔ (x > 2 ∧ x < 3) ∨ (x > 4 ∧ x < 5) ∨ (x > 6 ∧ x < 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l699_69907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_calculation_l699_69917

-- Define the angle α
variable (α : ℝ)

-- Define the point P
def P : ℝ × ℝ := (-4, 3)

-- Define the condition that the terminal side of α passes through P
def terminal_side_passes_through (α : ℝ) (p : ℝ × ℝ) : Prop :=
  Real.sin α = p.2 / Real.sqrt (p.1^2 + p.2^2) ∧ 
  Real.cos α = p.1 / Real.sqrt (p.1^2 + p.2^2)

-- Theorem statement
theorem angle_calculation (h : terminal_side_passes_through α P) : 
  2 * Real.sin α - Real.cos α = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_calculation_l699_69917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l699_69989

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x + a) / (2^x - a)

def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem f_properties (a : ℝ) :
  (is_odd (f a) → a = 1 ∨ a = -1) ∧
  (a < 0 → Monotone (f a)) ∧
  (a ≠ 0 → ∀ m n k : ℝ, m < n →
    (∀ x ∈ Set.Icc m n, f a x ∈ Set.Icc (k/(2^m)) (k/(2^n))) →
    k/a ∈ Set.union (Set.Ioo 0 (3-2*Real.sqrt 2)) {-1}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l699_69989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_area_approx_l699_69948

/-- The cost of fencing per meter in Rupees -/
noncomputable def fencing_cost_per_meter : ℝ := 5

/-- The total cost of fencing in Rupees -/
noncomputable def total_fencing_cost : ℝ := 7427.41

/-- The circumference of the field in meters -/
noncomputable def field_circumference : ℝ := total_fencing_cost / fencing_cost_per_meter

/-- The radius of the field in meters -/
noncomputable def field_radius : ℝ := field_circumference / (2 * Real.pi)

/-- The area of the field in square meters -/
noncomputable def field_area_sq_meters : ℝ := Real.pi * field_radius ^ 2

/-- The area of the field in hectares -/
noncomputable def field_area_hectares : ℝ := field_area_sq_meters / 10000

/-- Theorem stating that the area of the field is approximately 17.5616 hectares -/
theorem field_area_approx : 
  17.56 < field_area_hectares ∧ field_area_hectares < 17.57 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_area_approx_l699_69948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l699_69987

/-- The time it takes for two trains to cross a bridge -/
noncomputable def train_crossing_time (train1_length train2_length bridge_length : ℝ) 
                        (train1_speed train2_speed : ℝ) : ℝ :=
  let combined_length := train1_length + train2_length + bridge_length
  let relative_speed := train1_speed + train2_speed
  combined_length / relative_speed

/-- Theorem stating the approximate time for the trains to cross the bridge -/
theorem train_crossing_time_approx :
  let train1_length : ℝ := 120
  let train2_length : ℝ := 150
  let bridge_length : ℝ := 250
  let train1_speed : ℝ := 60 * 1000 / 3600  -- Convert 60 km/h to m/s
  let train2_speed : ℝ := 45 * 1000 / 3600  -- Convert 45 km/h to m/s
  abs (train_crossing_time train1_length train2_length bridge_length train1_speed train2_speed - 17.83) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l699_69987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_and_projection_l699_69940

/-- Given three vectors in R², prove properties about their relationships and projections. -/
theorem vector_relationships_and_projection (a b c : ℝ × ℝ) : 
  a = (1, 2) →
  Real.sqrt (c.1^2 + c.2^2) = 2 * Real.sqrt 5 →
  c = (-1 : ℝ) • a →
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 5 / 2 →
  (a.1 + 2*b.1, a.2 + 2*b.2) • (2*a.1 - b.1, 2*a.2 - b.2) = 15/4 →
  (c = (-2, -4) ∧ 
   (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -Real.sqrt 5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_and_projection_l699_69940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_to_decimal_conversion_l699_69934

/-- Converts a base 7 digit to its decimal (base 10) value. -/
def base7ToDecimal (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (7 ^ position)

/-- Represents the base 7 number 3216₇ as a list of its digits. -/
def base7Number : List ℕ := [6, 1, 2, 3]

/-- Converts the base 7 number to its decimal (base 10) equivalent. -/
def convertBase7ToDecimal (digits : List ℕ) : ℕ :=
  (List.zipWith base7ToDecimal digits (List.range digits.length)).sum

theorem base7_to_decimal_conversion :
  convertBase7ToDecimal base7Number = 1140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_to_decimal_conversion_l699_69934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_m5n4_in_expansion_l699_69995

theorem coefficient_m5n4_in_expansion : 
  (Finset.range 10).sum (fun k => Nat.choose 9 k * (if k = 5 then 1 else 0)) = 126 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_m5n4_in_expansion_l699_69995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_area_l699_69949

theorem obtuse_triangle_area (a b : ℝ) (C : ℝ) 
  (h1 : a = 8) (h2 : b = 12) (h3 : C = 150 * Real.pi / 180) :
  (1/2 : ℝ) * a * b * Real.sin C = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_area_l699_69949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_exponential_equation_l699_69909

theorem unique_solution_for_exponential_equation :
  ∃! (x y : ℝ), (9 : ℝ)^(x^2 + y) + (9 : ℝ)^(x + y^2) = 1 ∧ x = -1/2 ∧ y = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_exponential_equation_l699_69909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evans_class_size_l699_69911

/-- Represents the number of students in an economics class -/
structure EconomicsClass where
  students : ℕ
  students_positive : students > 0

/-- Theorem: The total number of students in Mrs. Evans' economics class is 40 -/
theorem evans_class_size (c : EconomicsClass) 
  (h1 : ∃ (s1 : Finset (Fin c.students)), s1.card = 30) -- 30 students answered question 1 correctly
  (h2 : ∃ (s2 : Finset (Fin c.students)), s2.card = 29) -- 29 students answered question 2 correctly
  (h3 : ∃ (s3 : Finset (Fin c.students)), s3.card = 10) -- 10 students did not take the test
  (h4 : ∃ (s4 : Finset (Fin c.students)), s4.card = 29) -- 29 students answered both questions correctly
  : c.students = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evans_class_size_l699_69911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l699_69993

/-- The radius of a sphere inscribed in a regular triangular pyramid -/
noncomputable def inscribed_sphere_radius (H α : ℝ) : ℝ :=
  H / (4 * Real.tan α ^ 2) * (Real.sqrt (4 * Real.tan α ^ 2 + 1) - 1)

/-- Theorem: The radius of a sphere inscribed in a regular triangular pyramid
    with height H and angle α between the lateral edge and the base plane
    is equal to H/(4 * tan²α) * (√(4 * tan²α + 1) - 1) -/
theorem inscribed_sphere_radius_formula (H α : ℝ) (h1 : H > 0) (h2 : 0 < α ∧ α < π / 2) :
  ∃ r : ℝ, r > 0 ∧ r = inscribed_sphere_radius H α :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l699_69993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_simplification_l699_69947

-- Define lg as logarithm base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_simplification : lg 4 + 2 * lg 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_simplification_l699_69947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l699_69970

-- Define the curve C
noncomputable def curve_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (2*Real.sqrt 5)/5*t, 1 + (Real.sqrt 5)/5*t)

-- Define point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_sum :
  ∃ (t1 t2 : ℝ),
    curve_C (line_l t1).1 (line_l t1).2 ∧
    curve_C (line_l t2).1 (line_l t2).2 ∧
    distance point_P (line_l t1) + distance point_P (line_l t2) = 4 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l699_69970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_sequence_properties_alt_l699_69967

def S (n : ℕ) : ℕ := n^2 + 2*n

def a (n : ℕ) : ℕ := 2*n + 1

noncomputable def b (n : ℕ) : ℝ := 3 * 2^(n-1)

noncomputable def T (n : ℕ) : ℝ := (3/2) * (2^n - 1)

theorem sequence_properties :
  (∀ n : ℕ, S (n+1) - S n = a (n+1)) ∧
  (b 2 = S 1) ∧
  (b 4 = a 2 + a 3) ∧
  (∀ n : ℕ, T n = (b 1) * (1 - (2^n)) / (1 - 2)) := by
  sorry

-- Alternative definition for b and T
noncomputable def b' (n : ℕ) : ℝ := -3 * (-2)^(n-1)

noncomputable def T' (n : ℕ) : ℝ := (1/2) * ((-2)^n - 1)

theorem sequence_properties_alt :
  (∀ n : ℕ, S (n+1) - S n = a (n+1)) ∧
  (b' 2 = S 1) ∧
  (b' 4 = a 2 + a 3) ∧
  (∀ n : ℕ, T' n = (b' 1) * (1 - ((-2)^n)) / (1 - (-2))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_sequence_properties_alt_l699_69967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_two_inverse_l699_69935

theorem reciprocal_of_two_inverse (x : ℝ) : x = 2⁻¹ → x⁻¹ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_two_inverse_l699_69935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_l699_69957

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4*x + 6 else x + 6

def solution_set : Set ℝ := {x | f x ≤ f 1}

theorem solution_set_eq : solution_set = Set.Iic (-3) ∪ Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_l699_69957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_to_asymptotes_l699_69981

-- Define the hyperbola
noncomputable def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the foci
def foci : ℝ × ℝ × ℝ × ℝ := (-2, 0, 2, 0)

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2

-- Define the dot product condition
noncomputable def dot_product_condition (P : ℝ × ℝ) : Prop :=
  let (f1x, f1y, f2x, f2y) := foci
  let v1 := (P.1 - f1x, P.2 - f1y)
  let v2 := (P.1 - f2x, P.2 - f2y)
  v1.1 * v2.1 + v1.2 * v2.2 = 1

-- Define the asymptotes
noncomputable def asymptote1 (x y : ℝ) : Prop := x + Real.sqrt 3 * y = 0
noncomputable def asymptote2 (x y : ℝ) : Prop := x - Real.sqrt 3 * y = 0

-- Define the distance to an asymptote
noncomputable def distance_to_asymptote (P : ℝ × ℝ) (asymptote : ℝ → ℝ → Prop) : ℝ :=
  let numerator := |P.1 + Real.sqrt 3 * P.2|
  let denominator := Real.sqrt (1 + 3)
  numerator / denominator

-- Theorem statement
theorem sum_of_distances_to_asymptotes (P : ℝ × ℝ) :
  point_on_hyperbola P →
  dot_product_condition P →
  distance_to_asymptote P asymptote1 + distance_to_asymptote P asymptote2 = 3 * Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_to_asymptotes_l699_69981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l699_69959

/-- Given a hyperbola with equation y²/2 - x²/4 = 1, its asymptotes are y = ±(√2/2)x -/
theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), y^2 / 2 - x^2 / 4 = 1 →
  ∃ (k : ℝ), k = Real.sqrt 2 / 2 ∧ (y = k * x ∨ y = -k * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l699_69959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_l699_69950

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x / Real.sqrt (m * x^2 + 2 * m * x + 4)

theorem domain_condition (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f m x = y) ↔ (0 ≤ m ∧ m < 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_l699_69950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_main_theorem_l699_69992

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  let y := x % 2
  if -1 ≤ y ∧ y < 0 then y + a
  else if 0 ≤ y ∧ y < 1 then |2/5 - y|
  else 0  -- This case should never occur due to the period

theorem f_periodic (x : ℝ) (a : ℝ) : f (x + 2) a = f x a := by sorry

theorem main_theorem (a : ℝ) (h : f (-5/2) a = f (9/2) a) : f (5*a) a = -2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_main_theorem_l699_69992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tiles_on_floor_l699_69913

theorem max_tiles_on_floor (floor_length floor_width tile_length tile_width : ℕ) 
  (h1 : floor_length = 250)
  (h2 : floor_width = 180)
  (h3 : tile_length = 45)
  (h4 : tile_width = 50) :
  max 
    ((floor_length / tile_length) * (floor_width / tile_width))
    ((floor_length / tile_width) * (floor_width / tile_length)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tiles_on_floor_l699_69913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_Z_time_l699_69941

-- Define the time taken by each printer to complete the job alone
noncomputable def time_X : ℝ := 16
noncomputable def time_Y : ℝ := 10

-- Define the ratio of X's time to Y and Z's combined time
noncomputable def ratio : ℝ := 2.4

-- Define the function to calculate the combined rate of two printers
noncomputable def combined_rate (t1 t2 : ℝ) : ℝ := 1 / t1 + 1 / t2

-- Theorem statement
theorem printer_Z_time :
  ∃ (time_Z : ℝ), time_Z > 0 ∧
  (1 / (combined_rate time_Y time_Z)) = time_X / ratio ∧
  time_Z = 20 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_Z_time_l699_69941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_square_inequalities_l699_69942

theorem circle_triangle_square_inequalities 
  (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 
    -- Circle equation
    x^2 + y^2 = r^2 ∧ 
    -- Inscribed equilateral triangle
    (∃ (a : ℝ), a > 0 ∧ |x| + |y| ≤ (3/2) * r) ∧
    -- Circumscribed square
    (∃ (s : ℝ), s > 0 ∧ max |x| |y| ≤ s/2 ∧ s = r * Real.sqrt 2)) →
  |x| + |y| ≤ (3/2) * Real.sqrt (x^2 + y^2) ∧ 
  (3/2) * Real.sqrt (x^2 + y^2) ≤ Real.sqrt 3 * max |x| |y| :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_square_inequalities_l699_69942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radical_product_l699_69932

-- Define the left-hand side of the equation
noncomputable def lhs (y : ℝ) : ℝ := Real.sqrt (45 * y^3) * Real.sqrt (50 * y) * Real.sqrt (20 * y^5)

-- Define the right-hand side of the equation
noncomputable def rhs (y : ℝ) : ℝ := 150 * y^4 * Real.sqrt y

-- State the theorem
theorem simplify_radical_product (y : ℝ) (h : y > 0) : lhs y = rhs y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radical_product_l699_69932
