import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_collinear_l722_72239

noncomputable section

-- Define basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define necessary functions and predicates
def dist (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

def Line (p q : Point) : Set Point :=
  {r : Point | ∃ t : ℝ, r = ⟨(1 - t) * p.x + t * q.x, (1 - t) * p.y + t * q.y⟩}

def Tangent (c : Circle) (p : Point) : Set Point :=
  {q : Point | dist p c.center = c.radius ∧ (∀ r ∈ Line p q, dist r c.center ≥ c.radius)}

def perpendicular (l1 l2 : Set Point) : Prop :=
  ∃ p : Point, p ∈ l1 ∧ p ∈ l2 ∧ ∀ q ∈ l1, ∀ r ∈ l2, (q.x - p.x) * (r.x - p.x) + (q.y - p.y) * (r.y - p.y) = 0

def collinear (p q r : Point) : Prop :=
  ∃ t : ℝ, q = ⟨(1 - t) * p.x + t * r.x, (1 - t) * p.y + t * r.y⟩

-- Main theorem
theorem circle_tangent_collinear
  (O : Circle) (A K M Q P L : Point)
  (h1 : dist A K = 2 * O.radius)
  (h2 : M ∈ {p : Point | dist p O.center < O.radius} \ Line A K)
  (h3 : Q ∈ {p : Point | dist p O.center = O.radius} ∩ Line A M)
  (h4 : P ∈ Tangent O Q ∩ {p : Point | perpendicular (Line p M) (Line A K)})
  (h5 : L ∈ {p : Point | dist p O.center = O.radius} ∩ Tangent O P) :
  collinear K L M :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_collinear_l722_72239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_line_segment_l722_72264

def endpoint1 : ℂ := -8 + 5*Complex.I
def endpoint2 : ℂ := 6 - 9*Complex.I

theorem midpoint_of_line_segment :
  (endpoint1 + endpoint2) / 2 = -1 - 2*Complex.I := by
  -- Expand the definition of endpoint1 and endpoint2
  simp [endpoint1, endpoint2]
  -- Perform the arithmetic
  ring
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_line_segment_l722_72264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_alpha_l722_72234

noncomputable def f (ω θ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + θ)

theorem f_value_at_alpha (ω θ α : ℝ) 
  (h_ω : ω > 0)
  (h_θ : 0 < θ ∧ θ < Real.pi)
  (h_period : ∀ x, f ω θ (x + Real.pi / ω) = f ω θ x)
  (h_odd : ∀ x, f ω θ (-x) + f ω θ x = 0)
  (h_tan : Real.tan α = 2) :
  f ω θ α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_alpha_l722_72234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beetles_movement_leaves_empty_square_l722_72265

/-- Represents a 5x5 chessboard -/
def Chessboard : Type := Fin 5 × Fin 5

/-- Represents the color of a square on the chessboard -/
inductive Color
| Black
| White

/-- Returns the color of a square on the chessboard -/
def squareColor (pos : Chessboard) : Color :=
  match (pos.1 + pos.2) % 2 with
  | 0 => Color.Black
  | _ => Color.White

/-- Represents a beetle on the chessboard -/
structure Beetle where
  position : Chessboard

/-- Represents the state of the chessboard with beetles -/
structure BoardState where
  beetles : Finset Beetle

/-- Returns true if two positions are adjacent -/
def isAdjacent (pos1 pos2 : Chessboard) : Prop :=
  (pos1.1 = pos2.1 ∧ (pos1.2 = pos2.2 + 1 ∨ pos1.2 + 1 = pos2.2)) ∨
  (pos1.2 = pos2.2 ∧ (pos1.1 = pos2.1 + 1 ∨ pos1.1 + 1 = pos2.1))

/-- Theorem: After beetles move, there will be at least one empty square -/
theorem beetles_movement_leaves_empty_square
  (initial : BoardState)
  (h_initial_count : initial.beetles.card = 25)
  (h_initial_distinct : ∀ b1 b2 : Beetle, b1 ∈ initial.beetles → b2 ∈ initial.beetles → b1 ≠ b2 → b1.position ≠ b2.position)
  (final : BoardState)
  (h_movement : ∀ b : Beetle, b ∈ initial.beetles → ∃ b' : Beetle, b' ∈ final.beetles ∧ isAdjacent b.position b'.position)
  : ∃ pos : Chessboard, pos ∉ (final.beetles.toSet.image Beetle.position) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beetles_movement_leaves_empty_square_l722_72265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_has_winning_strategy_l722_72231

/-- Represents a position on the board -/
structure Position where
  x : Fin 11
  y : Fin 11

/-- Represents a wall on the board -/
structure Wall where
  x : Fin 12
  y : Fin 12
  horizontal : Bool

/-- Represents the game state -/
structure GameState where
  chipPosition : Position
  walls : List Wall
  turn : Nat

/-- Represents a move by Petya -/
inductive PetyaMove
  | Up
  | Down
  | Left
  | Right

/-- Represents a move by Vasya -/
def VasyaMove := Wall

/-- Checks if a position is on the edge of the board -/
def isOnEdge (pos : Position) : Bool :=
  pos.x = 0 || pos.x = 10 || pos.y = 0 || pos.y = 10

/-- Applies moves to the game state -/
def applyMoves (game : GameState) (petyaMove : PetyaMove) (vasyaMove : VasyaMove) : GameState :=
  sorry

/-- The main theorem stating that Vasya has a winning strategy -/
theorem vasya_has_winning_strategy :
  ∃ (strategy : GameState → VasyaMove),
    ∀ (game : GameState) (petyaMove : PetyaMove),
      let newGame := applyMoves game petyaMove (strategy game)
      ¬isOnEdge newGame.chipPosition := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_has_winning_strategy_l722_72231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l722_72280

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Function to switch between players -/
def Player.other : Player → Player
| First => Second
| Second => First

/-- Represents the state of the quadratic equation -/
structure QuadraticState where
  a : ℝ
  b : ℝ
  c : ℝ
  current_player : Player

/-- Represents a move in the game -/
structure Move where
  coefficient : ℝ
  position : Fin 3

/-- Apply a move to the current state -/
def apply_move (state : QuadraticState) (move : Move) : QuadraticState :=
  match move.position with
  | 0 => { state with a := move.coefficient, current_player := state.current_player.other }
  | 1 => { state with b := move.coefficient, current_player := state.current_player.other }
  | 2 => { state with c := move.coefficient, current_player := state.current_player.other }

/-- Check if the quadratic equation has real roots -/
def has_real_roots (state : QuadraticState) : Prop :=
  state.b^2 - 4*state.a*state.c ≥ 0

/-- The main theorem: Second player can always ensure a win -/
theorem second_player_wins :
  ∀ (initial_state : QuadraticState),
  initial_state.a ≠ 0 →
  ∃ (strategy : QuadraticState → Move),
  ∀ (game : List Move),
  let final_state := game.foldl apply_move initial_state
  has_real_roots final_state :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l722_72280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l722_72200

theorem power_of_three (m n : ℝ) (h1 : (3 : ℝ)^m = 4) (h2 : (3 : ℝ)^n = 5) : 
  (3 : ℝ)^(m-2*n) = 4/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l722_72200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_dependent_on_time_s_depends_on_t_l722_72288

-- Define the variables and constants
variable (g : ℝ) -- acceleration due to gravity
variable (t : ℝ) -- time
variable (s : ℝ) -- distance

-- Define the formula
noncomputable def distance_formula (g t : ℝ) : ℝ := (g * t^2) / 2

-- State the theorem
theorem distance_dependent_on_time (g : ℝ) :
  ∃ (f : ℝ → ℝ), ∀ t, f t = distance_formula g t :=
by
  -- Define f as the function that calculates distance given time
  let f := λ t : ℝ ↦ distance_formula g t
  -- Show that this f satisfies the required property
  use f
  -- For all t, f t equals the distance formula
  intro t
  -- This is true by definition of f
  rfl

-- The dependent variable s is determined by the independent variable t
theorem s_depends_on_t (g : ℝ) :
  ∃ (f : ℝ → ℝ), ∀ t, s = f t :=
by
  -- Use the previous theorem
  obtain ⟨f, hf⟩ := distance_dependent_on_time g
  -- This f is the function we're looking for
  use f
  -- We need to show that for all t, s = f t
  -- But we don't have enough information to prove this
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_dependent_on_time_s_depends_on_t_l722_72288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l722_72232

open Real

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the function h
def h (t : ℝ) : ℝ → ℝ := fun x ↦ f (x + t)

-- State the theorem
theorem range_of_m :
  ∀ t : ℝ, t ∈ Set.Ioo 0 π →
  (∃ y : ℝ, ∀ x : ℝ, h t x = h t (-π/3 - x)) →
  (∀ x ∈ Set.Icc (π/4) (π/2), |f x - m| < 3) →
  m ∈ Set.Ioo (-1) 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l722_72232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_zero_l722_72218

theorem sin_sum_zero : Real.sin (π - 2) + Real.sin (3 * π + 2) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_zero_l722_72218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sculpture_painting_cost_l722_72212

/-- Paint type with cost per kilogram and coverage per kilogram -/
structure Paint where
  cost_per_kg : ℝ
  coverage_per_kg : ℝ

/-- Calculate the cost of painting a given area with a specific paint -/
noncomputable def paint_cost (p : Paint) (area : ℝ) : ℝ :=
  (area / p.coverage_per_kg) * p.cost_per_kg

/-- The total cost of painting the sculpture -/
noncomputable def total_painting_cost (paint_a paint_b paint_c : Paint) : ℝ :=
  paint_cost paint_a 50 + paint_cost paint_b 80 + paint_cost paint_c 30

/-- Theorem stating that the total cost of painting the sculpture is $374 -/
theorem sculpture_painting_cost :
  let paint_a : Paint := { cost_per_kg := 60, coverage_per_kg := 20 }
  let paint_b : Paint := { cost_per_kg := 45, coverage_per_kg := 25 }
  let paint_c : Paint := { cost_per_kg := 80, coverage_per_kg := 30 }
  total_painting_cost paint_a paint_b paint_c = 374 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sculpture_painting_cost_l722_72212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_alpha_l722_72296

theorem cos_two_alpha (α : ℝ) 
  (h1 : Real.sin (π/4 - α) = 3/5)
  (h2 : -π/4 < α)
  (h3 : α < 0) : 
  Real.cos (2*α) = 24/25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_alpha_l722_72296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dependency_and_determinant_l722_72201

def vector1 : Fin 3 → ℝ := ![1, 2, 4]
def vector2 (k : ℝ) : Fin 3 → ℝ := ![3, k, 6]

theorem vector_dependency_and_determinant (k : ℝ) :
  (∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧ (c1 • vector1 + c2 • vector2 k = 0)) ↔ k = 6
  ∧
  k ≠ 6 → Matrix.det !![vector1 0, vector2 k 0; vector1 1, vector2 k 1] = k - 6 :=
by
  sorry

#check vector_dependency_and_determinant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dependency_and_determinant_l722_72201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contains_nonzero_lattice_point_l722_72257

-- Define a convex region in ℝ²
def ConvexRegion (R : Set (ℝ × ℝ)) : Prop := sorry

-- Define symmetry about the origin
def SymmetricAboutOrigin (R : Set (ℝ × ℝ)) : Prop := 
  ∀ p : ℝ × ℝ, p ∈ R ↔ (-p.1, -p.2) ∈ R

-- Define the area of a region
noncomputable def Area (R : Set (ℝ × ℝ)) : ℝ := sorry

-- Define a lattice point
def LatticePoint (p : ℝ × ℝ) : Prop := 
  Int.fract p.1 = 0 ∧ Int.fract p.2 = 0

-- The main theorem
theorem contains_nonzero_lattice_point (R : Set (ℝ × ℝ)) 
  (hConvex : ConvexRegion R) 
  (hSymmetric : SymmetricAboutOrigin R) 
  (hArea : Area R > 4) : 
  ∃ p : ℝ × ℝ, p ∈ R ∧ p ≠ (0, 0) ∧ LatticePoint p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contains_nonzero_lattice_point_l722_72257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_size_l722_72225

/-- A structure representing a set M with 4n+3 elements and its subsets -/
structure SetSystem (n : ℕ) where
  M : Finset ℕ
  A : Fin (4*n+3) → Finset ℕ
  h_card_M : M.card = 4*n+3
  h_subset : ∀ i, A i ⊆ M
  h_n_plus_one : ∀ S : Finset ℕ, S ⊆ M → S.card = n+1 → 
    ∃! i, S ⊆ A i
  h_size : ∀ i, (A i).card ≥ 2*n+1

/-- The main theorem stating that the intersection of any two distinct subsets has exactly n elements -/
theorem intersection_size (n : ℕ) (s : SetSystem n) :
  ∀ i j, i < j → (s.A i ∩ s.A j).card = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_size_l722_72225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_successful_purchase_probability_l722_72259

/-- The probability of a successful purchase of football tickets -/
theorem successful_purchase_probability
  (m n : ℕ) -- m: number of people with 50 yuan bills, n: number of people with 100 yuan bills
  (h : m ≥ n) -- condition that m is greater than or equal to n
  : ℚ -- The probability is a rational number
  :=
(m - n + 1 : ℚ) / (m + 1 : ℚ)

#check successful_purchase_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_successful_purchase_probability_l722_72259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_problem_l722_72254

noncomputable section

/-- Calculate compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

/-- Calculate simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem simple_interest_problem (principal : ℝ) :
  let ci := compound_interest 4000 10 2
  let si := simple_interest principal 8 2
  si = ci / 2 → principal = 2625 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_problem_l722_72254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_area_ratio_l722_72248

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := 
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs ((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)))

/-- A line y = d - 2x with 1 < d < 8 intersects the y-axis at P and the line x = 4 at S.
    The ratio of the area of triangle QRS to the area of triangle QOP is 1:4. -/
theorem line_intersection_area_ratio (d : ℝ) 
  (h1 : 1 < d) (h2 : d < 8)
  (h3 : ∃ P : ℝ × ℝ, P.1 = 0 ∧ P.2 = d)  -- P on y-axis
  (h4 : ∃ S : ℝ × ℝ, S.1 = 4 ∧ S.2 = d - 2 * 4)  -- S on x = 4
  (h5 : ∃ Q : ℝ × ℝ, Q.1 = d / 2 ∧ Q.2 = 0)  -- Q on x-axis
  (h6 : ∃ R : ℝ × ℝ, R.1 = 4 ∧ R.2 = 0)  -- R on x-axis and x = 4
  (h7 : ∃ O : ℝ × ℝ, O = (0, 0))  -- Origin
  (h8 : (area_triangle Q R S) / (area_triangle Q O P) = 1 / 4) :
  d = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_area_ratio_l722_72248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_term_is_3755_l722_72284

-- Define the sequence
def our_sequence (n : ℕ) : ℕ :=
  sorry  -- Definition omitted, to be filled based on the sequence description

-- Theorem statement
theorem fiftieth_term_is_3755 : our_sequence 50 = 3755 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_term_is_3755_l722_72284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_sum_squared_factorials_l722_72274

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_squared_factorials (n : ℕ) : ℕ := 
  (List.range n).map (λ i => (factorial (i + 1))^2) |>.sum

theorem last_two_digits_of_sum_squared_factorials : 
  sum_of_squared_factorials 5 % 100 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_sum_squared_factorials_l722_72274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_ploughing_problem_l722_72297

/-- The number of hectares the farmer needed to plough per day to finish on time -/
noncomputable def required_hectares_per_day (total_area actual_rate extra_days remaining_area : ℝ) : ℝ :=
  let actual_days := (total_area - remaining_area) / actual_rate
  let planned_days := actual_days - extra_days
  total_area / planned_days

theorem farmer_ploughing_problem 
  (total_area : ℝ) 
  (actual_rate : ℝ) 
  (extra_days : ℝ) 
  (remaining_area : ℝ) 
  (h1 : total_area = 3780)
  (h2 : actual_rate = 85)
  (h3 : extra_days = 2)
  (h4 : remaining_area = 40) :
  required_hectares_per_day total_area actual_rate extra_days remaining_area = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_ploughing_problem_l722_72297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_infinite_n_rephinado_set_l722_72238

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- δ as defined in the problem -/
noncomputable def δ : ℝ := φ + 1

/-- Definition of n-th residue modulo p -/
def is_nth_residue (n p a : ℕ) : Prop :=
  ∃ x : ℕ, x^n % p = a % p

/-- Definition of n-rephinado prime -/
def is_n_rephinado_prime (n p : ℕ) : Prop :=
  Nat.Prime p ∧
  n ∣ (p - 1) ∧
  ∀ a : ℕ, a ≤ ⌊(p : ℝ) ^ (1 / δ)⌋ → is_nth_residue n p a

/-- The set of n for which there are infinitely many n-rephinado primes -/
def infinite_n_rephinado_set : Set ℕ :=
  {n : ℕ | Set.Infinite {p : ℕ | is_n_rephinado_prime n p}}

/-- The main theorem -/
theorem finite_infinite_n_rephinado_set : Set.Finite infinite_n_rephinado_set := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_infinite_n_rephinado_set_l722_72238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_possible_D_l722_72250

/-- The sum of interior angles of a hexagon in degrees -/
noncomputable def hexagon_angle_sum : ℝ := 720

/-- The number of interior angles in a hexagon -/
def num_angles : ℕ := 6

/-- The average measure of each interior angle in a hexagon -/
noncomputable def avg_angle : ℝ := hexagon_angle_sum / num_angles

/-- Represents the common difference of the arithmetic sequence of interior angles -/
def common_difference : ℝ → ℝ := λ d ↦ d

/-- Represents the smallest angle in the arithmetic sequence -/
noncomputable def smallest_angle (d : ℝ) : ℝ := avg_angle - (5/2) * common_difference d

/-- Represents the largest angle in the arithmetic sequence -/
noncomputable def largest_angle (d : ℝ) : ℝ := avg_angle + (5/2) * common_difference d

/-- Theorem: The greatest possible value of D is 24 degrees -/
theorem greatest_possible_D : 
  ∃ D : ℝ, D = 24 ∧ 
  ∀ d : ℝ, 
    (0 < smallest_angle d ∧ largest_angle d < 180) → 
    abs (common_difference d) < D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_possible_D_l722_72250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l722_72244

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 4 - y^2 / 16 = 1

-- Define the center, left focus, and eccentricity of C
def center : ℝ × ℝ := (0, 0)
noncomputable def left_focus : ℝ × ℝ := (-2 * Real.sqrt 5, 0)
noncomputable def eccentricity : ℝ := Real.sqrt 5

-- Define the left and right vertices of C
def left_vertex : ℝ × ℝ := (-2, 0)
def right_vertex : ℝ × ℝ := (2, 0)

-- Define a line passing through (-4, 0)
def line_through_neg_four (m : ℝ) (x y : ℝ) : Prop := x = m * y - 4

-- Define the intersection point P
def intersection_point (P : ℝ × ℝ) (m : ℝ) : Prop :=
  ∃ (M N : ℝ × ℝ),
    hyperbola_C M.1 M.2 ∧
    hyperbola_C N.1 N.2 ∧
    line_through_neg_four m M.1 M.2 ∧
    line_through_neg_four m N.1 N.2 ∧
    M.2 > 0 ∧ M.1 < 0 ∧  -- M is in the second quadrant
    (P.2 - left_vertex.2) / (P.1 - left_vertex.1) = (M.2 - left_vertex.2) / (M.1 - left_vertex.1) ∧
    (P.2 - right_vertex.2) / (P.1 - right_vertex.1) = (N.2 - right_vertex.2) / (N.1 - right_vertex.1)

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola_C x y ↔ x^2 / 4 - y^2 / 16 = 1) ∧
  (∀ m P, intersection_point P m → P.1 = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l722_72244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valley_function_m_range_min_L_value_l722_72263

-- Define valley function and valley interval
def is_valley_function (f : ℝ → ℝ) (a b x₀ : ℝ) : Prop :=
  a < x₀ ∧ x₀ < b ∧
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ x₀ → f y < f x) ∧
  (∀ x y, x₀ ≤ x ∧ x < y ∧ y ≤ b → f x < f y)

-- Part 2
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x - m * Real.log (x - 1)

theorem valley_function_m_range :
  ∀ m > 0, is_valley_function (f m) 2 4 x₀ ↔ 2 < m ∧ m < 18 :=
sorry

-- Part 3
def h (p q : ℝ) (x : ℝ) : ℝ := -x^4 + p*x^3 + q*x^2 + (4 - 3*p - 2*q)*x

noncomputable def L (p q : ℝ) : ℝ := Real.sqrt ((9*p^2)/16 + (3*p)/2 - 3 + 2*q)

theorem min_L_value :
  (∀ p q : ℝ, h p q 1 ≤ h p q 2 ∧ h p q 1 ≤ 0 →
    ∃ a b x₀ : ℝ, is_valley_function (h p q) a b x₀) →
  (∃ p₀ q₀ : ℝ, ∀ p q : ℝ, L p q ≥ L p₀ q₀ ∧ L p₀ q₀ = Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valley_function_m_range_min_L_value_l722_72263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l722_72216

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

noncomputable def combined_work_rate (rate_a rate_b : ℝ) : ℝ := rate_a + rate_b

noncomputable def days_to_complete (combined_rate : ℝ) : ℝ := 1 / combined_rate

theorem work_completion_time 
  (days_a days_b : ℝ) 
  (ha : days_a = 9) 
  (hb : days_b = 18) : 
  days_to_complete (combined_work_rate (work_rate days_a) (work_rate days_b)) = 6 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l722_72216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_sin_squared_l722_72240

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2

-- State the theorem
theorem center_of_symmetry_sin_squared :
  ∃ (c : ℝ × ℝ), c = (π / 4, 1 / 2) ∧
  ∀ (x : ℝ), f (c.fst + x) = f (c.fst - x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_sin_squared_l722_72240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_l722_72294

theorem same_terminal_side (θ : Real) : 
  (∃ k : Int, θ = -7 * Real.pi / 8 + 2 * k * Real.pi) ↔ 
  (∃ t : Real, Real.cos θ = Real.cos (-7 * Real.pi / 8) ∧ Real.sin θ = Real.sin (-7 * Real.pi / 8)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_l722_72294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l722_72266

theorem trigonometric_problem (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos α = 4/5)
  (h4 : Real.cos (α + β) = 3/5) :
  Real.sin β = 7/25 ∧ 2*α + β = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l722_72266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_squares_l722_72219

/-- The number of squares of side length n in an 8x8 chessboard -/
def squaresOfSize (n : Nat) : Nat :=
  (9 - n) * (9 - n)

/-- The total number of squares in an 8x8 chessboard -/
def totalSquares : Nat :=
  Finset.sum (Finset.range 8) (fun n => squaresOfSize (n + 1))

theorem chessboard_squares :
  totalSquares = 204 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_squares_l722_72219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_l722_72271

open InnerProductSpace NormedSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cosine_of_angle (u v : V) 
  (hu : ‖u‖ = 5)
  (hv : ‖v‖ = 7)
  (huv : ‖u + v‖ = 9) :
  inner u v / (‖u‖ * ‖v‖) = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_l722_72271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_worth_calculation_l722_72270

/-- Represents the total worth of the stock in Rupees -/
def total_worth : ℝ → Prop := λ x => x > 0

/-- The profit percentage for the first portion of the stock -/
def profit_percent_1 : ℝ := 0.1

/-- The loss percentage for the second portion of the stock -/
def loss_percent_2 : ℝ := 0.05

/-- The loss percentage for the third portion of the stock -/
def loss_percent_3 : ℝ := 0.1

/-- The portion of stock sold at profit -/
def stock_portion_1 : ℝ := 0.3

/-- The portion of stock sold at first loss -/
def stock_portion_2 : ℝ := 0.4

/-- The portion of stock sold at second loss -/
def stock_portion_3 : ℝ := 0.3

/-- The overall loss in Rupees -/
def overall_loss : ℝ := 500

theorem stock_worth_calculation (x : ℝ) (h : total_worth x) :
  (stock_portion_1 * profit_percent_1 * x) -
  (stock_portion_2 * loss_percent_2 * x) -
  (stock_portion_3 * loss_percent_3 * x) = -overall_loss →
  x = 25000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_worth_calculation_l722_72270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_lovers_who_disliked_l722_72226

/-- Proves that the number of meat lovers who disliked the new menu is 50 --/
theorem meat_lovers_who_disliked (total : ℕ) (liked : ℕ) (disliked : ℕ) 
  (vegan_percent : ℚ) (vegetarian_percent : ℚ) (gluten_free_percent : ℚ)
  (lactose_intolerant_percent : ℚ) (low_sodium_percent : ℚ) (pescatarian_percent : ℚ) :
  total = 500 →
  liked = 300 →
  disliked = 200 →
  vegan_percent = 25 / 100 →
  vegetarian_percent = 20 / 100 →
  gluten_free_percent = 15 / 100 →
  lactose_intolerant_percent = 35 / 100 →
  low_sodium_percent = 30 / 100 →
  pescatarian_percent = 10 / 100 →
  (disliked : ℚ) * (1 - (lactose_intolerant_percent + low_sodium_percent + pescatarian_percent)) = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_lovers_who_disliked_l722_72226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_angle_difference_l722_72287

theorem regular_polygon_angle_difference : 
  ∃ (s : Finset (ℕ × ℕ)), 
    (∀ (n k : ℕ), (n, k) ∈ s ↔ 
      n > k ∧ k ≥ 3 ∧ ((n - 2) * 180 / n - (k - 2) * 180 / k : ℚ) = 1) ∧
    s.card = 52 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_angle_difference_l722_72287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_six_when_k_is_one_k_equals_one_when_series_sum_is_six_l722_72252

noncomputable def series_term (n : ℕ) (k : ℝ) : ℝ := (2 + n * k) / (2^n)

noncomputable def series_sum (k : ℝ) : ℝ := 2 + ∑' n, series_term n k

theorem series_sum_equals_six_when_k_is_one :
  series_sum 1 = 6 := by sorry

theorem k_equals_one_when_series_sum_is_six :
  ∀ k : ℝ, series_sum k = 6 → k = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_six_when_k_is_one_k_equals_one_when_series_sum_is_six_l722_72252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_proof_l722_72235

theorem x_value_proof (x y : ℝ) : 
  (7 : ℝ)^(x - y) = 343 → (7 : ℝ)^(x + y) = 16807 → x = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_proof_l722_72235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial15_trailing_zeros_base81_l722_72299

/-- The number of trailing zeros in the base-81 representation of a natural number -/
def trailingZerosBase81 (n : ℕ) : ℕ := sorry

/-- 15 factorial -/
def factorial15 : ℕ := Nat.factorial 15

theorem factorial15_trailing_zeros_base81 : 
  trailingZerosBase81 factorial15 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial15_trailing_zeros_base81_l722_72299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cows_on_farm_l722_72247

/-- Represents the number of cows in a single herd -/
def X : ℕ := sorry

/-- Represents the total number of herds on the farm -/
def H : ℕ := sorry

/-- The number of cows in half of the herds -/
def half_herd_cows : ℕ := 2800

/-- The theorem stating that the total number of cows on the farm is 5600 -/
theorem total_cows_on_farm : 
  (H / 2 : ℕ) * X = half_herd_cows → X > 0 → H > 0 → H * X = 5600 := by
  sorry

#check total_cows_on_farm

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cows_on_farm_l722_72247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_rating_inequality_l722_72281

noncomputable def judge (i : ℕ) (x : Fin a) : Prop := sorry
def passes (x : Prop) : Prop := x

theorem competition_rating_inequality (a b k : ℕ) : 
  b ≥ 3 → 
  Odd b → 
  (∀ (i j : ℕ), i < b → j < b → i ≠ j → 
    (∃ (S : Finset (Fin a)), S.card ≤ k ∧ 
      (∀ x ∈ S, (judge i x ↔ judge j x)))) →
  (k : ℚ) / a ≥ (b - 1 : ℚ) / (2 * b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_rating_inequality_l722_72281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_consecutive_zeros_l722_72221

theorem sqrt_two_consecutive_zeros (k : ℕ) :
  ∀ t : ℕ, (∃ d : ℕ → ℕ, (Real.sqrt 2 = 1 + ∑' i, (d i : ℝ) / 10^(i+1)) ∧
    (∀ i ∈ Finset.range k, d (t + i + 1) = 0) ∧
    (d t ≠ 0)) →
  t + 1 > k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_consecutive_zeros_l722_72221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baker_remaining_cakes_l722_72267

/-- Given that a baker initially made 48 cakes and sold 44 cakes, 
    prove that the number of cakes remaining is 4. -/
theorem baker_remaining_cakes
  (initial_cakes : Int)
  (sold_cakes : Int)
  (h1 : initial_cakes = 48)
  (h2 : sold_cakes = 44) :
  initial_cakes - sold_cakes = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_baker_remaining_cakes_l722_72267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maddie_hair_color_boxes_l722_72241

def makeup_palette_price : ℚ := 15
def lipstick_price : ℚ := 5/2
def hair_color_price : ℚ := 4
def total_spent : ℚ := 67
def num_makeup_palettes : ℕ := 3
def num_lipsticks : ℕ := 4

def hair_color_boxes : ℕ := 
  (((total_spent - (num_makeup_palettes * makeup_palette_price + num_lipsticks * lipstick_price)) / hair_color_price).floor).toNat

theorem maddie_hair_color_boxes : hair_color_boxes = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maddie_hair_color_boxes_l722_72241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_ratio_sum_l722_72258

/-- Represents a checkerboard with its dimensions and line counts. -/
structure Checkerboard where
  size : Nat
  horizontalLines : Nat
  verticalLines : Nat

/-- Calculates the total number of rectangles on the checkerboard. -/
def totalRectangles (board : Checkerboard) : Nat :=
  (board.horizontalLines.choose 2) * (board.verticalLines.choose 2)

/-- Calculates the total number of squares on the checkerboard. -/
def totalSquares (board : Checkerboard) : Nat :=
  (board.size * (board.size + 1) * (2 * board.size + 1)) / 6

/-- The main theorem about the ratio of squares to rectangles on the checkerboard. -/
theorem checkerboard_ratio_sum (board : Checkerboard) 
    (h1 : board.size = 8)
    (h2 : board.horizontalLines = 9)
    (h3 : board.verticalLines = 9) :
    ∃ (m n : Nat), 
      (totalSquares board : Rat) / (totalRectangles board) = m / n ∧ 
      Nat.Coprime m n ∧
      m + n = 125 := by
  sorry

#eval totalRectangles ⟨8, 9, 9⟩
#eval totalSquares ⟨8, 9, 9⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_ratio_sum_l722_72258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_equal_polygons_with_non_coincident_vertices_l722_72255

/-- A polygon in a plane -/
structure Polygon where
  vertices : Set (ℝ × ℝ)
  is_polygon : Prop  -- Changed from IsPolygon to Prop

/-- Two polygons are equal if they have the same area and perimeter -/
noncomputable def equal_polygons (F F' : Polygon) : Prop :=
  ∃ (area perimeter : Polygon → ℝ),
    area F = area F' ∧ perimeter F = perimeter F'

/-- All vertices of one polygon belong to another polygon -/
def vertices_belong (F F' : Polygon) : Prop :=
  ∀ v ∈ F.vertices, v ∈ F'.vertices ∨ ∃ (point_inside : Polygon → (ℝ × ℝ) → Prop), point_inside F' v

/-- Theorem: There exist two equal polygons where all vertices of one belong to the other,
    but not all vertices coincide -/
theorem exist_equal_polygons_with_non_coincident_vertices :
  ∃ (F F' : Polygon),
    equal_polygons F F' ∧
    vertices_belong F F' ∧
    ¬(F.vertices = F'.vertices) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_equal_polygons_with_non_coincident_vertices_l722_72255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grass_field_width_l722_72278

/-- Proves that the width of a rectangular grass field is 180 m given specific conditions --/
theorem grass_field_width :
  ∀ (w : ℝ),
  (65 + 2 * 2.5) * (w + 2 * 2.5) - 65 * w = 1250 →
  w = 180 := by
  intro w hypothesis
  -- Simplify the left side of the equation
  have : 70 * (w + 5) - 65 * w = 1250 := by
    calc
      70 * (w + 5) - 65 * w = (65 + 2 * 2.5) * (w + 2 * 2.5) - 65 * w := by ring
      _ = 1250 := hypothesis
  
  -- Expand and simplify
  have : 5 * w = 900 := by
    calc
      5 * w = 70 * w + 350 - 65 * w - 350 := by ring
      _ = 70 * (w + 5) - 65 * w - 350 := by ring
      _ = 1250 - 350 := by rw [this]
      _ = 900 := by ring

  -- Solve for w
  have : w = 180 := by
    calc
      w = (5 * w) / 5 := by ring
      _ = 900 / 5 := by rw [this]
      _ = 180 := by norm_num

  exact this


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grass_field_width_l722_72278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l722_72272

/-- The line y = x + 1 -/
def line (x : ℝ) : ℝ := x + 1

/-- The circle (x - 3)^2 + y^2 = 1 -/
def circleEq (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1

/-- The length of the tangent from a point (a, line a) to the circle -/
noncomputable def tangent_length (a : ℝ) : ℝ := Real.sqrt (2 * a^2 - 4 * a + 9)

/-- The minimum length of the tangent from the line to the circle is √7 -/
theorem min_tangent_length :
  ∃ a : ℝ, ∀ b : ℝ, tangent_length a ≤ tangent_length b ∧ tangent_length a = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l722_72272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l722_72293

noncomputable def f (a ω x : ℝ) : ℝ := a * Real.sin (ω * x) - Real.cos (ω * x)

theorem function_properties (a ω m : ℝ) (h_a_pos : a > 0) (h_ω_pos : ω > 0) 
  (h_m_pos : m > 0) (h_m_nat : ∃ n : ℕ, m = n) (h_max_value : ∀ x, f a ω x ≤ 2) 
  (h_symmetry : ∀ x, f a ω x = f a ω (2 * π / m - x)) 
  (h_ω_min : ∃ n : ℕ, ω = n ∧ ∀ k : ℕ, 0 < k → k < n → ¬(∀ x, f a k x = f a k (2 * π / m - x))) :
  a = Real.sqrt 3 ∧ (∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < 10 ∧ 
    f a ω x₁ = 2 ∧ f a ω x₂ = 2 ∧ f a ω x₃ = 2 ∧
    ∀ x, 0 < x ∧ x < 10 → f a ω x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l722_72293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unitsDigit_17_2023_unitsDigit_power_7_l722_72291

-- Define the function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the function for the units digit pattern of powers of 7
def unitsDigitPower7 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | _ => 3

-- State the theorem
theorem unitsDigit_17_2023 :
  unitsDigit (17^2023) = 3 := by
  sorry

-- Additional lemma to establish the relationship between 17^n and 7^n
lemma unitsDigit_17_eq_7 (n : ℕ) :
  unitsDigit (17^n) = unitsDigit (7^n) := by
  sorry

-- Theorem to prove the pattern of units digits for powers of 7
theorem unitsDigit_power_7 (n : ℕ) :
  unitsDigit (7^n) = unitsDigitPower7 n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unitsDigit_17_2023_unitsDigit_power_7_l722_72291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_production_and_daily_B_l722_72243

/-- Represents the daily production of type B masks in millions -/
noncomputable def daily_production_B : ℝ := sorry

/-- Represents the number of type A masks produced in millions -/
noncomputable def type_A_production : ℝ := sorry

/-- Represents the number of type B masks produced in millions -/
noncomputable def type_B_production : ℝ := sorry

/-- Theorem stating the optimal production and daily production of type B masks -/
theorem optimal_production_and_daily_B :
  -- Daily production of type A is twice that of type B
  (2 * daily_production_B = daily_production_B * 2) →
  -- Relationship between production time of A and B masks
  (50 / daily_production_B - 40 / (2 * daily_production_B) = 6) →
  -- Total production constraint
  (type_A_production + type_B_production = 200) →
  -- Time constraint
  (type_A_production / (2 * daily_production_B) + type_B_production / daily_production_B ≤ 30) →
  -- Profit calculation
  (let profit := 0.8 * type_A_production + 1.2 * type_B_production
   ∀ other_A other_B,
     other_A + other_B = 200 →
     other_A / (2 * daily_production_B) + other_B / daily_production_B ≤ 30 →
     0.8 * other_A + 1.2 * other_B ≤ profit) →
  -- Conclusion
  daily_production_B = 5 ∧ type_A_production = 100 ∧ type_B_production = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_production_and_daily_B_l722_72243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l722_72237

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(-x^2 + 3*x + 2)

theorem f_monotone_increasing (a : ℝ) (h : 0 < a ∧ a < 1) :
  StrictMonoOn (f a) (Set.Ioi (3/2 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l722_72237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l722_72229

open Real

theorem triangle_inequalities (A B C : ℝ) (h : A + B + C = π) :
  (sin A * sin (A/2) + sin B * sin (B/2) + sin C * sin (C/2) ≤ 4 / sqrt 3) ∧
  (sin A * cos (A/2) + sin B * cos (B/2) + sin C * cos (C/2) ≤ 4 / sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l722_72229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_complement_l722_72282

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x^2)

-- Define the domain of f
def M : Set ℝ := {x : ℝ | 1 - x^2 ≥ 0}

-- State the theorem
theorem domain_complement :
  (Set.univ : Set ℝ) \ M = {x : ℝ | x < -1 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_complement_l722_72282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l722_72286

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then 1/x - 1 else -2*x + a

theorem range_of_a :
  {a : ℝ | ∀ x₁ x₂, x₁ ≠ x₂ → f a x₁ ≠ f a x₂} = Set.Ici 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l722_72286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_formula_l722_72211

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  -- The length of the longer base
  a : ℝ
  -- The length of the shorter base
  b : ℝ
  -- Assumption that a > b > 0
  h_a_gt_b : a > b
  h_b_pos : b > 0

/-- The length of MN in an isosceles trapezoid with an inscribed circle -/
noncomputable def length_MN (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ :=
  (2 * t.a * t.b) / (t.a + t.b)

/-- Theorem stating that the length of MN is 2ab / (a + b) -/
theorem length_MN_formula (t : IsoscelesTrapezoidWithInscribedCircle) :
  length_MN t = (2 * t.a * t.b) / (t.a + t.b) := by
  -- Unfold the definition of length_MN
  unfold length_MN
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_formula_l722_72211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l722_72275

noncomputable def sign (x : ℝ) : ℝ :=
  if x < 0 then -1
  else if x > 0 then 1
  else 0

def satisfies_equations (x y z : ℝ) : Prop :=
  x = 2023 - 2024 * sign (y + z) ∧
  y = 2023 - 2024 * sign (x + z) ∧
  z = 2023 - 2024 * sign (x + y)

theorem exactly_three_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ)),
    solutions.card = 3 ∧
    ∀ (x y z : ℝ), (x, y, z) ∈ solutions ↔ satisfies_equations x y z :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l722_72275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_count_l722_72230

/-- Represents a point on a 6x6 geoboard -/
structure GeoPoint where
  x : Fin 6
  y : Fin 6

/-- Represents a segment on the geoboard -/
structure GeoSegment where
  start : GeoPoint
  finish : GeoPoint

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : GeoPoint) : Nat :=
  (p1.x.val - p2.x.val)^2 + (p1.y.val - p2.y.val)^2

/-- Checks if a triangle is isosceles -/
def isIsosceles (a b c : GeoPoint) : Prop :=
  squaredDistance a b = squaredDistance a c ∨
  squaredDistance a b = squaredDistance b c ∨
  squaredDistance a c = squaredDistance b c

/-- The main theorem -/
theorem isosceles_triangle_count (de : GeoSegment) :
  de.start.y = de.finish.y →  -- DE is horizontal
  squaredDistance de.start de.finish = 9 →  -- DE = 3 units
  (∃! (n : Nat), ∃ (points : Finset GeoPoint),
    points.card = n ∧
    (∀ f ∈ points, f ≠ de.start ∧ f ≠ de.finish) ∧
    (∀ f ∈ points, isIsosceles de.start de.finish f) ∧
    (∀ f : GeoPoint, f ≠ de.start ∧ f ≠ de.finish →
      isIsosceles de.start de.finish f → f ∈ points) ∧
    n = 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_count_l722_72230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_sum_property_l722_72207

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * (Real.cos x + Real.sqrt 3 * Real.sin x) - Real.sqrt 3 + 1

theorem f_increasing_and_sum_property :
  (∀ x₁ x₂ : ℝ, -π/12 < x₁ ∧ x₁ < x₂ ∧ x₂ < 5*π/12 → f x₁ < f x₂) ∧
  (∀ x₁ x₂ x₃ : ℝ, x₁ ∈ Set.Icc (π/3) (π/2) ∧ x₂ ∈ Set.Icc (π/3) (π/2) ∧ x₃ ∈ Set.Icc (π/3) (π/2) →
    f x₁ + f x₂ - f x₃ > 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_sum_property_l722_72207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_when_m_neg_six_m_value_when_intersecting_line_l722_72277

-- Define the curve equation
def curve_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem 1: When m = -6, the center is (1, 2) and radius is √11
theorem circle_properties_when_m_neg_six :
  ∃ (x0 y0 r : ℝ), 
    (∀ x y, curve_equation x y (-6) ↔ (x - x0)^2 + (y - y0)^2 = r^2) ∧
    x0 = 1 ∧ y0 = 2 ∧ r = Real.sqrt 11 := by
  sorry

-- Theorem 2: When the circle intersects the line at M and N with |MN| = 4/√5, m = 4
theorem m_value_when_intersecting_line :
  ∃ (m xM yM xN yN : ℝ),
    curve_equation xM yM m ∧
    curve_equation xN yN m ∧
    line_equation xM yM ∧
    line_equation xN yN ∧
    distance xM yM xN yN = 4 / Real.sqrt 5 →
    m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_when_m_neg_six_m_value_when_intersecting_line_l722_72277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_solution_sum_correct_l722_72292

/-- The infinite geometric series representation of the original equation -/
noncomputable def infiniteSeries (x : ℝ) : ℝ := 1 + x - x^2 + x^3 - x^4 + x^5 - (x / (1 + x))

/-- The theorem stating the unique solution to the equation -/
theorem unique_solution :
  ∃! x : ℝ, x = infiniteSeries x ∧ |x| < 1 ∧ x = (-1 + Real.sqrt 5) / 2 := by
  sorry

/-- The sum of all real values of x that satisfy the equation -/
noncomputable def solution_sum : ℝ := (-1 + Real.sqrt 5) / 2

/-- Theorem proving that the solution_sum is correct -/
theorem solution_sum_correct :
  solution_sum = (-1 + Real.sqrt 5) / 2 ∧
  solution_sum = infiniteSeries solution_sum ∧
  |solution_sum| < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_solution_sum_correct_l722_72292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_odd_ones_l722_72215

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- A four-digit number represented as a list of digits -/
def FourDigitNumber := List Nat

/-- Check if a number has an odd number of 1s -/
def hasOddNumberOfOnes (n : FourDigitNumber) : Bool :=
  (n.filter (· = 1)).length % 2 = 1

/-- Generate all valid four-digit numbers -/
noncomputable def allFourDigitNumbers : List FourDigitNumber :=
  [1, 2, 3, 4, 5].bind (λ a =>
    digits.toList.bind (λ b =>
      digits.toList.bind (λ c =>
        digits.toList.map (λ d => [a, b, c, d]))))

/-- The main theorem -/
theorem count_numbers_with_odd_ones : 
  (allFourDigitNumbers.filter hasOddNumberOfOnes).length = 454 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_odd_ones_l722_72215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kids_2004_kids_2005_kids_2006_cookout_ratio_l722_72283

/-- Represents the number of kids at the cookout in a given year -/
def kids_at_cookout (year : ℕ) : ℕ := 
  match year with
  | 2004 => 60
  | 2005 => 30
  | 2006 => 20
  | _ => 0

/-- The number of kids at the cookout in 2004 was 60 -/
theorem kids_2004 : kids_at_cookout 2004 = 60 := rfl

/-- The number of kids at the cookout in 2005 was half of 2004 -/
theorem kids_2005 : kids_at_cookout 2005 = kids_at_cookout 2004 / 2 := rfl

/-- The number of kids at the cookout in 2006 was 20 -/
theorem kids_2006 : kids_at_cookout 2006 = 20 := rfl

/-- The ratio of kids in 2006 to 2005 is 2:3 -/
theorem cookout_ratio : 
  (kids_at_cookout 2006 : ℚ) / (kids_at_cookout 2005 : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kids_2004_kids_2005_kids_2006_cookout_ratio_l722_72283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sessions_needed_proof_l722_72204

def hamburgers_per_session : ℝ := 27.5
def total_hamburgers : ℕ := 578
def cooked_hamburgers : ℕ := 163

noncomputable def sessions_needed : ℕ :=
  let remaining_hamburgers := total_hamburgers - cooked_hamburgers
  let sessions_float := (remaining_hamburgers : ℝ) / hamburgers_per_session
  Int.toNat (Int.ceil sessions_float)

#check sessions_needed = 16

-- Proof
theorem sessions_needed_proof : sessions_needed = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sessions_needed_proof_l722_72204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l722_72268

/-- Parabola type representing y^2 = 2x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.equation point.1 point.2

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_focus_distance 
  (p : Parabola) 
  (h_eq : p.equation = fun x y => y^2 = 2*x) 
  (h_focus : p.focus = (1/2, 0)) 
  (P : PointOnParabola p) 
  (h_x : P.point.1 = 2) : 
  distance P.point p.focus = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l722_72268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l722_72290

theorem triangle_cosine_inequality (A B C : ℝ) :
  (1 / (1 + Real.cos B ^ 2 + Real.cos C ^ 2)) +
  (1 / (1 + Real.cos C ^ 2 + Real.cos A ^ 2)) +
  (1 / (1 + Real.cos A ^ 2 + Real.cos B ^ 2)) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l722_72290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_value_l722_72262

def b : ℕ → ℚ
  | 0 => 4  -- We need to define b for 0 to cover all natural numbers
  | 1 => 4
  | 2 => 5
  | n+3 => b (n+2) / b (n+1)

theorem b_2023_value : b 2023 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_value_l722_72262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_river_distance_l722_72220

/-- Proves that a boat traveling on a river with a given current speed, downstream time, and upstream time covers a specific distance downstream. -/
theorem boat_river_distance 
  (current_speed : ℝ) 
  (downstream_time upstream_time : ℝ) 
  (h_current : current_speed = 1) 
  (h_downstream : downstream_time = 4) 
  (h_upstream : upstream_time = 6) : 
  let boat_speed := (upstream_time + downstream_time) * current_speed / (upstream_time - downstream_time)
  let downstream_distance := (boat_speed + current_speed) * downstream_time
  downstream_distance = 24 := by
  sorry

#check boat_river_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_river_distance_l722_72220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_angles_l722_72202

/-- 
  Given a parallelogram with sides a and b, and diagonals c and f,
  if a^4 + b^4 = c^2 * f^2, then the angles of the parallelogram are 45° and 135°.
-/
theorem parallelogram_angles (a b c f : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ f > 0) 
  (h_eq : a^4 + b^4 = c^2 * f^2) : 
  ∃ (α β : ℝ), α = 45 * π / 180 ∧ β = 135 * π / 180 ∧ 
    c^2 = a^2 + b^2 - 2*a*b*(Real.cos α) ∧ 
    f^2 = a^2 + b^2 + 2*a*b*(Real.cos α) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_angles_l722_72202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_quadrilateral_area_l722_72208

/-- A circle in the xy-plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in the xy-plane --/
def Point := ℝ × ℝ

/-- The equation of a circle --/
def circleEquation (c : Circle) (p : Point) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- A line in the xy-plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point lies on a line --/
def pointOnLine (l : Line) (p : Point) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- A line is tangent to a circle --/
def isTangent (l : Line) (c : Circle) : Prop :=
  ∃ p : Point, circleEquation c p ∧ pointOnLine l p ∧
    ∀ q : Point, q ≠ p → circleEquation c q → ¬pointOnLine l q

/-- The area of a quadrilateral --/
noncomputable def quadrilateralArea (p q r s : Point) : ℝ :=
  sorry

theorem circle_tangent_quadrilateral_area :
  ∀ c : Circle,
  circleEquation c (1, 2) →
  circleEquation c (1, 0) →
  circleEquation c (0, 1) →
  ∃ l₁ l₂ : Line,
    isTangent l₁ c ∧
    isTangent l₂ c ∧
    pointOnLine l₁ (1, 2) ∧
    pointOnLine l₂ (1, 2) ∧
    quadrilateralArea (0, 0) (1, 0) (1, 2) (0, 5/4) = 13/8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_quadrilateral_area_l722_72208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_inequality_condition_l722_72224

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + Real.log x

-- Part 1: Tangent line perpendicular condition
theorem tangent_perpendicular (a : ℝ) :
  (∀ x > 0, ∃ y, y = f a x) →
  (let f' := λ x ↦ 2*x - 2*a + 1/x;
   f' 1 * (1/2) = -1) →
  a = 5/2 := by sorry

-- Part 2: Inequality condition
theorem inequality_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 (Real.exp 1), 2*x*(Real.log x) ≥ -x^2 + a*x - 3) →
  a ∈ Set.Iic 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_inequality_condition_l722_72224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_true_propositions_l722_72206

-- Define the propositions
def proposition1 (b : ℝ) : Prop := b^2 = 9 → b = 3

-- For proposition2, we'll use a simplified version that captures the essence without using undefined symbols
def proposition2 : Prop := ∃ A B : ℝ, A ≠ B ∧ ∃ f : ℝ → ℝ, f A = f B

def proposition3 (c : ℝ) : Prop := c ≤ 1 → ∃ x : ℝ, x^2 + 2*x + c = 0

def proposition4 {α : Type*} (A B : Set α) : Prop := ¬(B ⊆ A) → ¬(A ∪ B = A)

-- Theorem stating which propositions are true
theorem true_propositions :
  (∃ b : ℝ, ¬(proposition1 b)) ∧
  ¬proposition2 ∧
  (∀ c : ℝ, proposition3 c) ∧
  (∀ {α : Type*} (A B : Set α), proposition4 A B) :=
by
  sorry

#check true_propositions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_true_propositions_l722_72206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_l722_72203

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3}

-- Define B as a function of x
def B (x : ℕ) : Set ℕ := {1, x, 4}

-- Define the theorem
theorem x_value : ∃ x : ℕ, (A ∪ B x = {1, 2, 3, 4}) → (x = 2 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_l722_72203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_pentagon_area_l722_72223

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Area of a trapezoid given bases and height -/
noncomputable def trapezoidArea (base1 base2 height : ℝ) : ℝ :=
  (base1 + base2) * height / 2

/-- The y-coordinate of vertex C in a symmetric pentagon with area 60 -/
theorem symmetric_pentagon_area (p : Pentagon)
  (hA : p.A = (0, 0))
  (hB : p.B = (0, 6))
  (hC : p.C = (3, p.C.2))  -- y-coordinate is unknown
  (hD : p.D = (6, 6))
  (hE : p.E = (6, 0))
  (hSymmetry : p.C.1 = 3)  -- vertical line of symmetry
  (hArea : trapezoidArea 6 (p.C.2 - 6) 3 + 36 = 60) :
  p.C.2 = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_pentagon_area_l722_72223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2014_equals_one_fourth_l722_72273

def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 1/4
  | n+1 => 1 - 1/(mySequence n)

theorem sequence_2014_equals_one_fourth :
  mySequence 2013 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2014_equals_one_fourth_l722_72273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_halloween_cleanup_l722_72214

/-- The Halloween Cleanup Theorem -/
theorem halloween_cleanup (
  egg_cleanup_time : ℚ) 
  (tp_cleanup_time : ℚ)
  (total_cleanup_time : ℚ)
  (tp_rolls : ℕ)
  (h1 : egg_cleanup_time = 15 / 60)
  (h2 : tp_cleanup_time = 30)
  (h3 : total_cleanup_time = 225)
  (h4 : tp_rolls = 7) :
  (total_cleanup_time - tp_rolls * tp_cleanup_time) / egg_cleanup_time = 60 := by
  sorry

-- Remove the #eval line as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_halloween_cleanup_l722_72214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_sixth_l722_72279

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and terminal side passing through (-√3, 2), prove that tan(α - π/6) = -3√3 -/
theorem tan_alpha_minus_pi_sixth (α : Real) 
  (h1 : Real.tan α = -2 * Real.sqrt 3 / 3) : 
  Real.tan (α - Real.pi/6) = -3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_sixth_l722_72279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l722_72276

/-- The area of a quadrilateral formed by two right triangles sharing a hypotenuse -/
theorem quadrilateral_area (hypotenuse : ℝ) (angle1 : ℝ) (angle2 : ℝ) : 
  hypotenuse = 10 →
  angle1 = 30 * π / 180 →
  angle2 = 45 * π / 180 →
  (hypotenuse^2 * Real.sin angle1) / 4 + hypotenuse^2 / 4 = (25 * Real.sqrt 3 + 50) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l722_72276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l722_72233

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) := Real.sqrt (3 - 2*x - x^2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | -3 ≤ x ∧ x ≤ 1} = {x : ℝ | ∃ y : ℝ, f x = y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l722_72233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_positions_possible_l722_72222

/-- Circle represented by its equation in the form (x - h)^2 + (y - k)^2 = r^2 -/
structure Circle where
  h : ℝ  -- x-coordinate of the center
  k : ℝ  -- y-coordinate of the center
  r : ℝ  -- radius
  r_pos : r > 0

/-- The distance between two points in 2D space -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Possible relative positions of two circles -/
inductive CirclePosition
  | intersecting
  | tangent
  | oneInsideOther

/-- The main theorem statement -/
theorem circle_positions_possible (a : ℝ) : ∃ (C1 C2 : Circle), 
  (C1.h = 0 ∧ C1.k = 0 ∧ C1.r = Real.sqrt 2) ∧ 
  (C2.h = -1 ∧ C2.k = 1 ∧ C2.r^2 = a + 3) ∧
  (∃ (p : CirclePosition), 
    (p = CirclePosition.intersecting ∨ 
     p = CirclePosition.tangent ∨ 
     p = CirclePosition.oneInsideOther)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_positions_possible_l722_72222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawsuit_probability_comparison_l722_72289

theorem lawsuit_probability_comparison :
  let p_win_first : ℚ := 3/10
  let p_win_second : ℚ := 1/2
  let p_lose_first : ℚ := 1 - p_win_first
  let p_lose_second : ℚ := 1 - p_win_second
  let p_win_both : ℚ := p_win_first * p_win_second
  let p_lose_both : ℚ := p_lose_first * p_lose_second
  let difference : ℚ := p_lose_both - p_win_both
  let percentage_difference : ℚ := (difference / p_win_both) * 100
  ∃ (ε : ℚ), abs (percentage_difference - 400/3) < ε ∧ ε < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawsuit_probability_comparison_l722_72289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_box_height_l722_72253

/-- A rectangular box with two large spheres and eight small spheres. -/
structure SphereBox where
  width : ℝ
  length : ℝ
  height : ℝ
  large_sphere_radius : ℝ
  small_sphere_radius : ℝ
  large_sphere_count : ℕ
  small_sphere_count : ℕ

/-- Small spheres are tangent to three faces of the box. -/
def small_sphere_tangent_to_three_faces (box : SphereBox) (small_sphere : ℝ × ℝ × ℝ) : Prop :=
  sorry

/-- Large spheres are tangent to four smaller spheres. -/
def large_sphere_tangent_to_four_small_spheres (box : SphereBox) (large_sphere : ℝ × ℝ × ℝ) : Prop :=
  sorry

/-- The two large spheres are tangent to each other. -/
def large_spheres_tangent_to_each_other (box : SphereBox) : Prop :=
  sorry

/-- The configuration of spheres in the box satisfies the given conditions. -/
def valid_configuration (box : SphereBox) : Prop :=
  box.width = 6 ∧
  box.length = 6 ∧
  box.large_sphere_radius = 3 ∧
  box.small_sphere_radius = 1.5 ∧
  box.large_sphere_count = 2 ∧
  box.small_sphere_count = 8 ∧
  (∀ small_sphere, small_sphere_tangent_to_three_faces box small_sphere) ∧
  (∀ large_sphere, large_sphere_tangent_to_four_small_spheres box large_sphere) ∧
  large_spheres_tangent_to_each_other box

/-- The main theorem: the height of the box is 15 given the valid configuration. -/
theorem sphere_box_height (box : SphereBox) (h_valid : valid_configuration box) : box.height = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_box_height_l722_72253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_iff_a_eq_neg_two_l722_72251

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line x + y = 1 -/
noncomputable def slope_line1 : ℝ := -1

/-- The slope of the line ax + 2y = 0 -/
noncomputable def slope_line2 (a : ℝ) : ℝ := -a / 2

/-- The statement that a = -2 is both sufficient and necessary for perpendicularity -/
theorem perpendicular_iff_a_eq_neg_two (a : ℝ) :
  perpendicular slope_line1 (slope_line2 a) ↔ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_iff_a_eq_neg_two_l722_72251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_probabilities_l722_72269

-- Define the shooting percentages and initial probabilities
noncomputable def playerA_shooting_percentage : ℝ := 0.6
noncomputable def playerB_shooting_percentage : ℝ := 0.8
noncomputable def initial_probability : ℝ := 0.5

-- Define the probability that player B takes the second shot
noncomputable def prob_B_second_shot : ℝ := 0.6

-- Define the probability that player A takes the i-th shot
noncomputable def prob_A_ith_shot (i : ℕ) : ℝ := 1/3 + (1/6) * (2/5)^(i-1)

-- Define the expected number of times player A shoots in the first n shots
noncomputable def expected_A_shots (n : ℕ) : ℝ := (5/18) * (1 - (2/5)^n) + n/3

-- State the theorem
theorem basketball_game_probabilities :
  (prob_B_second_shot = initial_probability * (1 - playerA_shooting_percentage) + 
                        initial_probability * playerB_shooting_percentage) ∧
  (∀ i : ℕ, prob_A_ith_shot i = 1/3 + (1/6) * (2/5)^(i-1)) ∧
  (∀ n : ℕ, expected_A_shots n = (5/18) * (1 - (2/5)^n) + n/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_probabilities_l722_72269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_l722_72209

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  5 * x^2 - 4 * y^2 + 60 = 0

/-- The coordinates of a focus of the hyperbola -/
noncomputable def focus_coordinate : ℝ × ℝ := (0, 3 * Real.sqrt 3)

/-- Theorem stating the relationship between the hyperbola equation and its foci -/
theorem hyperbola_foci :
  ∀ (x y : ℝ), hyperbola_equation x y →
  (∃ (s : ℝ), s = 1 ∨ s = -1) →
  (x, y) = (0, s * 3 * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_l722_72209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_qingtuan_pricing_l722_72217

/-- Represents the wholesale and retail pricing of Qingtuan -/
structure QingtuanPricing where
  sesame_wholesale : ℝ
  meatfloss_wholesale : ℝ
  sesame_retail : ℝ
  boxes_sold : ℝ → ℝ
  profit : ℝ → ℝ

/-- Theorem stating the optimal pricing and profit for Qingtuan sales -/
theorem optimal_qingtuan_pricing (q : QingtuanPricing) :
  q.sesame_wholesale = 40 ∧
  q.meatfloss_wholesale = 30 ∧
  q.sesame_retail = 65 ∧
  q.profit q.sesame_retail = 1750 :=
  by
  have h1 : q.meatfloss_wholesale = q.sesame_wholesale - 10 := sorry
  have h2 : 800 / q.sesame_wholesale = 600 / q.meatfloss_wholesale := sorry
  have h3 : q.boxes_sold 50 = 100 := sorry
  have h4 : ∀ (x : ℝ), q.boxes_sold (x + 1) = q.boxes_sold x - 2 := sorry
  have h5 : ∀ (x : ℝ), q.profit x = (x - q.sesame_wholesale) * q.boxes_sold x := sorry
  have h6 : q.sesame_retail ≤ 65 := sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_qingtuan_pricing_l722_72217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_coefficient_binomial_expansion_l722_72213

theorem highest_coefficient_binomial_expansion :
  let n : ℕ := 9
  let expansion := fun (k : ℕ) => (n.choose k) * ((-1 : ℤ)^(n - k)).toNat
  ∀ k : ℕ, k ≤ n → expansion k ≤ 126 ∧ ∃ k₀ : ℕ, k₀ ≤ n ∧ expansion k₀ = 126 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_coefficient_binomial_expansion_l722_72213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_vertices_on_sphere_surface_area_l722_72228

-- Define a cube type
structure Cube where
  edge_length : ℝ

-- Define a sphere type
structure Sphere where
  radius : ℝ

-- Define a function to calculate the surface area of a sphere
noncomputable def sphere_surface_area (s : Sphere) : ℝ := 4 * Real.pi * s.radius^2

-- We need to define this function, but its implementation is not needed for the statement
def sphere_contains_cube_vertices (c : Cube) (s : Sphere) : Prop := sorry

-- Theorem statement
theorem cube_vertices_on_sphere_surface_area 
  (c : Cube) 
  (h1 : c.edge_length = 2) 
  (h2 : ∃ s : Sphere, sphere_contains_cube_vertices c s) : 
  ∃ s : Sphere, sphere_surface_area s = 12 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_vertices_on_sphere_surface_area_l722_72228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_index_after_100_days_l722_72242

/-- The stock index after n days, given an initial index and daily percentage increase -/
noncomputable def stock_index (initial_index : ℝ) (daily_increase : ℝ) (n : ℕ) : ℝ :=
  initial_index * (1 + daily_increase / 100) ^ n

/-- Rounds a real number to three decimal places -/
noncomputable def round_to_three_decimals (x : ℝ) : ℝ :=
  (⌊x * 1000 + 0.5⌋ : ℝ) / 1000

theorem stock_index_after_100_days :
  round_to_three_decimals (stock_index 2 0.02 100) = 2.041 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_index_after_100_days_l722_72242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_extreme_coins_68_100_l722_72246

/-- Represents a coin with a unique weight -/
structure Coin where
  weight : ℕ
deriving Inhabited

/-- Represents a weighing on a balance scale -/
def Weighing := Coin → Coin → Bool

/-- Finds the heaviest and lightest coins among a list of coins -/
def findExtremeCoins (coins : List Coin) (weighings : ℕ) : Option (Coin × Coin) :=
  sorry

/-- Theorem: It's possible to find the heaviest and lightest coins among 68 coins using 100 weighings -/
theorem find_extreme_coins_68_100 :
  ∀ (coins : List Coin),
    coins.length = 68 →
    (∀ i j, i ≠ j → (coins.get! i).weight ≠ (coins.get! j).weight) →
    ∃ (heaviest lightest : Coin),
      heaviest ∈ coins ∧
      lightest ∈ coins ∧
      (∀ c ∈ coins, c.weight ≤ heaviest.weight) ∧
      (∀ c ∈ coins, c.weight ≥ lightest.weight) ∧
      findExtremeCoins coins 100 = some (heaviest, lightest) := by
  sorry

#check find_extreme_coins_68_100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_extreme_coins_68_100_l722_72246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_terms_count_l722_72249

theorem expansion_terms_count : ℕ := by
  -- Define the number of terms in each factor
  let terms_first_factor : ℕ := 2
  let terms_second_factor : ℕ := 5

  -- Define the total number of terms in the expansion
  let total_terms : ℕ := terms_first_factor * terms_second_factor

  -- Theorem statement
  have : total_terms = 10 := by
    rfl  -- reflexivity proves this trivial equality

  exact total_terms

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_terms_count_l722_72249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_percent_l722_72261

/-- Calculate the profit percent from a car sale given the costs and selling price -/
theorem car_sale_profit_percent
  (purchase_price repair_cost taxes insurance selling_price : ℝ)
  (h_purchase : purchase_price = 42000)
  (h_repair : repair_cost = 13000)
  (h_taxes : taxes = 5000)
  (h_insurance : insurance = 8000)
  (h_selling : selling_price = 69900) :
  let total_cost := purchase_price + repair_cost + taxes + insurance
  let profit := selling_price - total_cost
  let profit_percent := (profit / total_cost) * 100
  abs (profit_percent - 2.794) < 0.001 := by
  sorry

#check car_sale_profit_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_percent_l722_72261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_vote_percentage_l722_72227

theorem james_vote_percentage (total_votes : ℕ) (additional_votes_needed : ℕ) : 
  total_votes = 2000 →
  additional_votes_needed = 991 →
  (↑(total_votes / 2 + 1 + additional_votes_needed) : ℚ) / total_votes > 1 / 2 →
  (↑(total_votes - additional_votes_needed - 1) : ℚ) / total_votes = 1 / 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_vote_percentage_l722_72227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_equivalence_l722_72260

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := (1/2) * (f x) + 3

-- Statement to prove
theorem transform_equivalence (x y : ℝ) :
  g f x = y ↔ ∃ (y' : ℝ), f x = y' ∧ y = (1/2) * y' + 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_equivalence_l722_72260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_addition_and_scalar_multiplication_l722_72256

def c : Fin 3 → ℝ := ![3, 5, -2]
def d : Fin 3 → ℝ := ![-1, 4, 3]

theorem vector_addition_and_scalar_multiplication :
  (2 : ℝ) • c + (1/2 : ℝ) • d = ![5.5, 12, -2.5] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_addition_and_scalar_multiplication_l722_72256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l722_72236

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt 3 * (((a + b) * (b + c) * (c + a)) ^ (1/3 : ℝ)) ≥ 2 * Real.sqrt (a * b + b * c + c * a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l722_72236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_difference_approximation_l722_72245

noncomputable section

/-- The circumference of the smaller circle in meters -/
def c₁ : ℝ := 268

/-- The circumference of the larger circle in meters -/
def c₂ : ℝ := 380

/-- The mathematical constant pi -/
def π : ℝ := Real.pi

/-- The radius of the smaller circle -/
noncomputable def r₁ : ℝ := c₁ / (2 * π)

/-- The radius of the larger circle -/
noncomputable def r₂ : ℝ := c₂ / (2 * π)

/-- The area of the smaller circle -/
noncomputable def A₁ : ℝ := π * r₁^2

/-- The area of the larger circle -/
noncomputable def A₂ : ℝ := π * r₂^2

/-- The difference between the areas of the larger and smaller circles -/
noncomputable def area_difference : ℝ := A₂ - A₁

theorem circle_area_difference_approximation :
  abs (area_difference - 5778.33) < 0.01 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_difference_approximation_l722_72245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_traversal_l722_72298

/-- Represents a move on the chessboard -/
inductive Move
| Right
| Up
| LowerLeft

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Nat) (y : Nat)

/-- Checks if a position is valid on an n × n chessboard -/
def isValidPosition (n : Nat) (pos : Position) : Prop :=
  pos.x < n ∧ pos.y < n

/-- Applies a move to a position -/
def applyMove (pos : Position) (move : Move) : Position :=
  match move with
  | Move.Right => ⟨pos.x + 1, pos.y⟩
  | Move.Up => ⟨pos.x, pos.y + 1⟩
  | Move.LowerLeft => ⟨pos.x - 1, pos.y - 1⟩

/-- Represents a sequence of moves on the chessboard -/
def ChessPath := List Move

/-- Checks if a path covers all squares on an n × n chessboard exactly once -/
def isValidPath (n : Nat) (path : ChessPath) : Prop :=
  path.length = n * n - 1 ∧
  (∀ pos : Position, isValidPosition n pos → 
    ∃! i : Nat, i < path.length ∧ 
      (path.take i).foldl applyMove ⟨0, 0⟩ = pos)

/-- The main theorem to be proved -/
theorem chessboard_traversal (n : Nat) (h : n ≥ 2) :
  (∃ path : ChessPath, isValidPath n path) ↔ 
  (∃ k : Nat, k > 0 ∧ (n = 3 * k ∨ n = 3 * k + 1)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_traversal_l722_72298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_expression_l722_72285

-- Define the sequence a_n
def a : ℕ → ℤ
  | 0 => 1
  | n + 1 => a n + 3

-- State the theorem
theorem a_expression (n : ℕ) : a n = 3 * (n + 1) - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_expression_l722_72285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_fourth_power_sum_l722_72205

theorem sin_cos_fourth_power_sum (φ : ℝ) (h : Real.cos (2 * φ) = 1 / 4) :
  Real.sin φ ^ 4 + Real.cos φ ^ 4 = 17 / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_fourth_power_sum_l722_72205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vector_length_l722_72295

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Line structure -/
structure Line where
  passes_through : ℝ × ℝ → Prop

/-- Point structure -/
structure Point where
  coords : ℝ × ℝ

/-- Vector structure -/
structure MyVector where
  components : ℝ × ℝ

/-- The main theorem -/
theorem parabola_vector_length 
  (C : Parabola)
  (F : Point)
  (L : Line)
  (P Q M : Point)
  (h1 : C.equation = fun x y => y^2 = 4*x)
  (h2 : L.passes_through F.coords)
  (h3 : L.passes_through P.coords)
  (h4 : L.passes_through Q.coords)
  (h5 : L.passes_through M.coords)
  (h6 : (M.coords.1 - F.coords.1, M.coords.2 - F.coords.2) = 
        (3 * (P.coords.1 - F.coords.1), 3 * (P.coords.2 - F.coords.2))) :
  let FP := (P.coords.1 - F.coords.1, P.coords.2 - F.coords.2)
  Real.sqrt (FP.1^2 + FP.2^2) = 4/3 ∨ Real.sqrt (FP.1^2 + FP.2^2) = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vector_length_l722_72295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_with_18_degree_exterior_angle_has_20_sides_l722_72210

/-- The number of sides of a regular polygon given the measure of its exterior angle in degrees -/
noncomputable def regular_polygon_sides (exterior_angle : ℝ) : ℝ :=
  360 / exterior_angle

theorem regular_polygon_with_18_degree_exterior_angle_has_20_sides :
  regular_polygon_sides 18 = 20 := by
  -- Unfold the definition of regular_polygon_sides
  unfold regular_polygon_sides
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_with_18_degree_exterior_angle_has_20_sides_l722_72210
