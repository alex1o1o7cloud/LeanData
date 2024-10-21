import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_volume_plus_icing_area_l290_29061

/-- Represents the edge length of the cubical cake in inches -/
noncomputable def edge_length : ℝ := 4

/-- Represents the volume of one piece of the cake (triangle B) -/
noncomputable def cake_volume : ℝ := (edge_length^2 / 2) * edge_length

/-- Represents the icing area on the top face for one piece (triangle B) -/
noncomputable def top_icing_area : ℝ := edge_length^2 / 2

/-- Represents the icing area on the right lateral face for one piece (triangle B) -/
noncomputable def lateral_icing_area : ℝ := (edge_length * edge_length) / 2

/-- Represents the total icing area for one piece (triangle B) -/
noncomputable def total_icing_area : ℝ := top_icing_area + lateral_icing_area

/-- Theorem stating that the sum of the volume and icing area of one piece (triangle B) is 48 -/
theorem cake_volume_plus_icing_area : cake_volume + total_icing_area = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_volume_plus_icing_area_l290_29061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l290_29074

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l290_29074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l290_29020

theorem least_number_with_remainder (n : ℕ) : 
  (∀ d ∈ ({7, 9, 18} : Set ℕ), n % d = 4) ∧ 
  (∀ m < n, ∃ d ∈ ({7, 9, 18} : Set ℕ), m % d ≠ 4) → 
  n = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l290_29020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_paths_from_P_to_Q_l290_29037

-- Define the points
inductive Point : Type
  | P | Q | R | S | T | U

-- Define the direct connections between points
def directConnection : Point → Point → Prop
  | Point.P, Point.R => True
  | Point.P, Point.S => True
  | Point.R, Point.T => True
  | Point.S, Point.T => True
  | Point.S, Point.U => True
  | Point.U, Point.T => True
  | Point.U, Point.Q => True
  | Point.T, Point.Q => True
  | _, _ => False

-- Define a path as a list of points
def PathList := List Point

-- A valid path is one where each consecutive pair of points has a direct connection
def validPath : PathList → Prop
  | [] => True
  | [_] => True
  | (p::q::rest) => directConnection p q ∧ validPath (q::rest)

-- Count the number of valid paths between two points
def countPaths (start finish : Point) : Nat :=
  sorry

-- Theorem stating that there are 4 distinct paths from P to Q
theorem four_paths_from_P_to_Q : 
  countPaths Point.P Point.Q = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_paths_from_P_to_Q_l290_29037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_equation_l290_29095

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 16

-- Define the line cutting the chord
def chord_line_eq (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define the diameter line
def diameter_line_eq (x y : ℝ) : Prop := 2*x + y - 3 = 0

-- Theorem statement
theorem diameter_equation :
  ∀ (x₀ y₀ : ℝ),
  circle_eq x₀ y₀ →
  chord_line_eq x₀ y₀ →
  (∃ (x₁ y₁ : ℝ), circle_eq x₁ y₁ ∧ chord_line_eq x₁ y₁ ∧
    ∃ (x_mid y_mid : ℝ), x_mid = (x₀ + x₁)/2 ∧ y_mid = (y₀ + y₁)/2 ∧
    diameter_line_eq x_mid y_mid) →
  diameter_line_eq x₀ y₀ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_equation_l290_29095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equiangular_hexagon_exists_l290_29025

/-- A hexagon is a closed polygon with 6 sides -/
structure Hexagon where
  sides : Fin 6 → ℝ
  is_closed : True  -- This is a simplification; actual closure condition would be more complex

/-- An equiangular hexagon is a hexagon where all interior angles are equal -/
structure EquiangularHexagon extends Hexagon where
  angles_equal : True  -- This is a simplification; actual condition would involve angle measures

/-- The theorem statement -/
theorem equiangular_hexagon_exists (n : ℕ) (hn : 0 < n) :
  ∃ (h : EquiangularHexagon), ∃ (σ : Equiv (Fin 6) (Fin 6)),
    (∀ i, h.sides i ∈ (Set.range (λ k : Fin 6 => (n : ℝ) + k.val + 1))) ∧
    (∀ i, h.sides (σ i) = (n : ℝ) + i.val + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equiangular_hexagon_exists_l290_29025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_tree_l290_29069

/-- Calculates the time (in seconds) it takes for a train to pass a stationary point. -/
noncomputable def train_passing_time (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (5 / 18)  -- Convert km/h to m/s
  length / speed_ms

/-- Theorem: A train 140 meters long, traveling at 63 km/hr, will take 12 seconds to pass a stationary point. -/
theorem train_passing_tree (length : ℝ) (speed_kmh : ℝ) 
    (h1 : length = 140) 
    (h2 : speed_kmh = 63) : 
  train_passing_time length speed_kmh = 12 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_passing_time 140 63

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_tree_l290_29069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_spiders_sufficient_and_necessary_l290_29064

/-- Represents a position on the grid -/
structure Position where
  x : Nat
  y : Nat
deriving Repr

/-- The size of the grid -/
def gridSize : Nat := 2019

/-- Represents the state of the game -/
structure GameState where
  flyPos : Position
  spider1Pos : Position
  spider2Pos : Position
deriving Repr

/-- Checks if a position is within the grid -/
def isValidPosition (pos : Position) : Prop :=
  pos.x ≤ gridSize ∧ pos.y ≤ gridSize

/-- Checks if two positions are adjacent -/
def isAdjacent (pos1 pos2 : Position) : Prop :=
  (Int.natAbs (pos1.x - pos2.x) + Int.natAbs (pos1.y - pos2.y)) = 1

/-- Checks if the fly is caught -/
def isCaught (state : GameState) : Prop :=
  state.flyPos = state.spider1Pos ∨ state.flyPos = state.spider2Pos

/-- Represents a valid move for either the fly or a spider -/
def isValidMove (fromPos toPos : Position) : Prop :=
  isValidPosition fromPos ∧ isValidPosition toPos ∧ isAdjacent fromPos toPos

/-- The main theorem stating that two spiders are sufficient and necessary -/
theorem two_spiders_sufficient_and_necessary :
  ∀ (initialState : GameState),
    (isValidPosition initialState.flyPos ∧
     isValidPosition initialState.spider1Pos ∧
     isValidPosition initialState.spider2Pos) →
    ∃ (strategy : GameState → Position × Position),
      ∀ (flyMove : Position),
        isValidMove initialState.flyPos flyMove →
        let newSpiderPositions := strategy initialState
        let newState : GameState := {
          flyPos := flyMove,
          spider1Pos := newSpiderPositions.1,
          spider2Pos := newSpiderPositions.2
        }
        isValidMove initialState.spider1Pos newSpiderPositions.1 ∧
        isValidMove initialState.spider2Pos newSpiderPositions.2 ∧
        (isCaught newState ∨
         ∃ (nextStrategy : GameState → Position × Position),
           ∀ (nextFlyMove : Position),
             isValidMove newState.flyPos nextFlyMove →
             isCaught { newState with flyPos := nextFlyMove }) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_spiders_sufficient_and_necessary_l290_29064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_supervisor_salary_unchanged_l290_29004

/-- Represents the salary structure of a factory -/
structure FactorySalary where
  num_workers : Nat
  worker_total_salary : ℝ
  supervisor_salary : ℝ
  average_salary : ℝ

/-- Represents the change in salary structure after supervisor replacement -/
def new_supervisor_salary (initial : FactorySalary) (new_average : ℝ) : ℝ :=
  (new_average * (initial.num_workers + 1 : ℝ) - initial.worker_total_salary)

/-- Theorem stating the new supervisor's salary remains the same -/
theorem new_supervisor_salary_unchanged 
  (initial : FactorySalary)
  (h1 : initial.num_workers = 8)
  (h2 : initial.average_salary = 430)
  (h3 : initial.supervisor_salary = 870)
  (h4 : new_supervisor_salary initial 430 = 870) : 
  new_supervisor_salary initial 430 = initial.supervisor_salary :=
by
  sorry

#eval new_supervisor_salary 
  { num_workers := 8
  , worker_total_salary := 3000
  , supervisor_salary := 870
  , average_salary := 430 } 430

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_supervisor_salary_unchanged_l290_29004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_probability_l290_29065

-- Define the jump length
noncomputable def jumpLength : ℝ := 2

-- Define the rectangle boundaries
noncomputable def xMin : ℝ := 0
noncomputable def xMax : ℝ := 6
noncomputable def yMin : ℝ := 0
noncomputable def yMax : ℝ := 6

-- Define the starting point
noncomputable def startX : ℝ := 2
noncomputable def startY : ℝ := 3

-- Define jump probabilities
noncomputable def probLeft : ℝ := 1/3
noncomputable def probRight : ℝ := 1/3
noncomputable def probUp : ℝ := 1/6
noncomputable def probDown : ℝ := 1/6

-- Function to calculate the probability of ending on a vertical side
noncomputable def probVerticalSide (x y : ℝ) : ℝ :=
  sorry

-- Theorem statement
theorem frog_jump_probability :
  probVerticalSide startX startY = 8/9 := by
  sorry

#check frog_jump_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_probability_l290_29065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_friday_speed_l290_29077

/-- Harry's running speeds throughout the week -/
def HarrysSpeeds (monday tuesday wednesday thursday friday : ℝ) : Prop :=
  tuesday = monday * (1 - 0.3) ∧
  wednesday = monday * (1 + 0.5) ∧
  thursday = wednesday ∧
  friday = thursday * (1 + 0.6)

/-- Theorem: Harry's speed on Friday given his initial speed and daily changes -/
theorem harry_friday_speed (monday : ℝ) :
  monday = 10 →
  ∃ tuesday wednesday thursday friday : ℝ,
    HarrysSpeeds monday tuesday wednesday thursday friday ∧
    friday = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_friday_speed_l290_29077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l290_29026

noncomputable def f (x : ℝ) : ℝ := Real.cos (3 * x) + 4

theorem period_of_f :
  ∃ (p : ℝ), p > 0 ∧ 
  (∀ (x : ℝ), f (x + p) = f x) ∧ 
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l290_29026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_conditions_l290_29035

open Complex

def M (a b : ℤ) : Set ℂ := {(a + 3 : ℂ) + (b^2 - 1 : ℂ) * I, 8}
def N (a b : ℤ) : Set ℂ := {3 * I, (a^2 - 1 : ℂ) + (b + 2 : ℂ) * I}

theorem intersection_conditions (a b : ℤ) :
  (M a b ∩ N a b).Subset (M a b) →
  (M a b ∩ N a b).Nonempty →
  ((a = -3 ∧ b = 2) ∨ (a = 3 ∧ b = -2)) :=
by
  sorry

#check intersection_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_conditions_l290_29035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l290_29043

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := 2 * n + 3

-- Define b_n
noncomputable def b (n : ℕ) : ℝ :=
  if n % 2 = 1 then a n - 6 else 2 * a n

-- Define S_n (sum of first n terms of a_n)
noncomputable def S (n : ℕ) : ℝ := (n + 4) * n

-- Define T_n (sum of first n terms of b_n)
noncomputable def T (n : ℕ) : ℝ :=
  if n % 2 = 0 then
    (n * (3 * n + 7)) / 2
  else
    (3 * n^2 + 5 * n - 10) / 2

-- Theorem statement
theorem arithmetic_sequence_property (n : ℕ) :
  (S 4 = 32) →
  (T 3 = 16) →
  (∀ (k : ℕ), k > 5 → T k > S k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l290_29043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_translation_l290_29009

/-- Given a quadratic function f(x) = -2x^2, prove that translating its vertex
    to the point (-3, 2) results in the function g(x) = -2(x + 3)^2 + 2 -/
theorem quadratic_translation :
  let f : ℝ → ℝ := λ x ↦ -2 * x^2
  let g : ℝ → ℝ := λ x ↦ -2 * (x + 3)^2 + 2
  ∀ x, f (x + 3) = g x - 2 :=
by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_translation_l290_29009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_transformation_l290_29052

-- Define the original quadratic equation
def original_equation (x : ℝ) : Prop := (x - 3)^2 - 5*(x - 3) = 0

-- Define the factored form of the equation
def factored_equation (x : ℝ) : Prop := (x - 3)*(x - 8) = 0

-- Define the solutions
def solution1 : ℝ := 3
def solution2 : ℝ := 8

-- Define the concept of transformation thinking
def transformation_thinking (eq1 eq2 : ℝ → Prop) : Prop :=
  ∀ x, eq1 x ↔ eq2 x

-- Theorem statement
theorem quadratic_equation_transformation :
  transformation_thinking original_equation factored_equation ∧
  factored_equation solution1 ∧
  factored_equation solution2 →
  (reflects_transformation_thinking : Prop) :=
by
  intro h
  sorry

-- Define what it means for a method to reflect transformation thinking
def reflects_transformation_thinking : Prop :=
  ∃ eq1 eq2 : ℝ → Prop, transformation_thinking eq1 eq2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_transformation_l290_29052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_100th_term_l290_29091

def sequenceProperty (a : ℕ → ℕ) : Prop :=
  a 1 = 1010 ∧ 
  a 2 = 1012 ∧ 
  ∀ n : ℕ, n ≥ 1 → a n + 2 * a (n + 1) + a (n + 2) = 5 * n

theorem sequence_100th_term (a : ℕ → ℕ) (h : sequenceProperty a) : a 100 = 1175 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_100th_term_l290_29091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_9_15_l290_29057

/-- Represents a clock with hour and minute hands -/
structure Clock where
  hour_angle : ℝ
  minute_angle : ℝ

/-- Calculates the smaller angle between two angles on a circle -/
noncomputable def smaller_angle (a b : ℝ) : ℝ :=
  min (abs (a - b)) (360 - abs (a - b))

/-- Calculates the position of the hour hand at a given time -/
noncomputable def hour_hand_position (hour minute : ℕ) : ℝ :=
  (hour % 12 + minute / 60 : ℝ) * 30

/-- Calculates the position of the minute hand at a given time -/
def minute_hand_position (minute : ℕ) : ℝ :=
  (minute : ℝ) * 6

/-- Theorem stating that the smaller angle between clock hands at 9:15 p.m. is 172.5° -/
theorem clock_angle_at_9_15 :
  let c : Clock := {
    hour_angle := hour_hand_position 21 15,
    minute_angle := minute_hand_position 15
  }
  smaller_angle c.hour_angle c.minute_angle = 172.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_9_15_l290_29057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l290_29007

noncomputable def g (x : ℝ) : ℝ := (3*x - 4)*(x + 1) / x

theorem inequality_solution :
  ∀ x : ℝ, x ≠ 0 → (g x ≥ 0 ↔ (x ∈ Set.Icc (-1) 0 ∪ Set.Ici (4/3))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l290_29007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_points_theorem_l290_29040

/-- The parabola y^2 = 12x -/
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (3, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- A point satisfies the conditions if it's on the parabola and its distance to the focus is 9 -/
def satisfies_conditions (p : ℝ × ℝ) : Prop :=
  parabola p.1 p.2 ∧ distance p focus = 9

theorem parabola_points_theorem :
  ∀ p : ℝ × ℝ, satisfies_conditions p ↔ p = (6, 6 * Real.sqrt 2) ∨ p = (6, -6 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_points_theorem_l290_29040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_cos6_l290_29084

theorem min_sin6_cos6 (x : ℝ) : Real.sin x ^ 6 + Real.cos x ^ 6 ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_cos6_l290_29084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l290_29038

/-- Circle C₁ with center (6,0) and radius 1 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 6)^2 + p.2^2 = 1}

/-- Circle C₂ with center (3,4) and radius 6 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = 36}

/-- The centers of the circles -/
def center₁ : ℝ × ℝ := (6, 0)
def center₂ : ℝ × ℝ := (3, 4)

/-- The radii of the circles -/
def radius₁ : ℝ := 1
def radius₂ : ℝ := 6

/-- Two circles are internally tangent if the distance between their centers
    equals the absolute difference of their radii -/
def internally_tangent (center₁ center₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  (center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2 = (r₂ - r₁)^2

/-- Theorem: C₁ and C₂ are internally tangent -/
theorem circles_internally_tangent :
  internally_tangent center₁ center₂ radius₁ radius₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l290_29038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_squared_minus_3A_l290_29087

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_A_squared_minus_3A : det (A^2 - 3 • A) = 88 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_squared_minus_3A_l290_29087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_two_l290_29098

theorem complex_expression_equals_two :
  -1^2 + Int.natAbs (-2) + ((-8 : ℝ) ^ (1/3 : ℝ)) + Real.sqrt ((-3)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_two_l290_29098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l290_29086

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (k * a.1 = b.1 ∧ k * a.2 = b.2)

theorem parallel_vectors_lambda (l : ℝ) :
  are_parallel (2, 5) (l, 4) → l = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l290_29086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_count_mod_1000_l290_29083

def string_length : ℕ := 18
def num_A : ℕ := 4
def num_B : ℕ := 6
def num_C : ℕ := 7
def num_D : ℕ := 1

def first_segment_length : ℕ := 5
def middle_segment_length : ℕ := 6
def last_segment_length : ℕ := 7

def permutation_count (x : ℕ) : ℕ :=
  Nat.choose first_segment_length (num_A - x) *
  Nat.choose middle_segment_length x *
  Nat.choose last_segment_length (x + 1)

def M : ℕ := Finset.sum (Finset.range (num_A + 1)) permutation_count

theorem permutation_count_mod_1000 :
  M % 1000 = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_count_mod_1000_l290_29083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_triangle_sum_l290_29000

-- Define the triangle vertices
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 2)
def C (m : ℝ) : ℝ × ℝ := (6*m, 0)

-- Define the dividing line
def dividing_line (m : ℝ) (x : ℝ) : ℝ := m * x

-- Helper function definitions
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

def sum_of_roots (f : ℝ → ℝ) : ℝ := sorry

-- State the theorem
theorem equal_area_triangle_sum (m : ℝ) : 
  (∃ x : ℝ, dividing_line m x = (C m).2 ∧ 
   (area_triangle A B (x, dividing_line m x) = 
    area_triangle (x, dividing_line m x) B (C m))) →
  (sum_of_roots (λ m ↦ 6*m^2 + 2*m - 2) = -1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_triangle_sum_l290_29000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_points_l290_29029

def X : Finset ℕ := {1, 2}
def Y : Finset ℕ := {1, 3, 4}

theorem number_of_points : Finset.card (Finset.product X Y) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_points_l290_29029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l290_29054

noncomputable def f (x : ℝ) := 7 * Real.sin (x - Real.pi/6)

theorem f_increasing_on_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < Real.pi/2 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l290_29054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_side_b_value_l290_29028

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  Real.sin (t.B / 2) = Real.sqrt 5 / 5 ∧
  t.a * t.c * Real.cos t.B = 6 ∧
  t.c + t.a = 8

-- Theorem 1: Area of the triangle
theorem triangle_area (t : Triangle) (h : triangle_conditions t) : 
  (1 / 2) * t.a * t.c * Real.sin t.B = 4 := by
  sorry

-- Theorem 2: Value of side b
theorem side_b_value (t : Triangle) (h : triangle_conditions t) :
  t.b = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_side_b_value_l290_29028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_lent_to_B_is_5000_l290_29042

/-- Represents the loan details and interest calculations --/
structure LoanDetails where
  rate : ℚ  -- Interest rate per annum
  timeB : ℚ  -- Time period for B's loan in years
  timeC : ℚ  -- Time period for C's loan in years
  amountC : ℚ  -- Amount lent to C
  totalInterest : ℚ  -- Total interest received from both B and C

/-- Calculates the amount lent to B given the loan details --/
def calculateAmountLentToB (loan : LoanDetails) : ℚ :=
  (loan.totalInterest - loan.rate * loan.amountC * loan.timeC) / (loan.rate * loan.timeB)

/-- Theorem stating that the amount lent to B is 5000 given the specified conditions --/
theorem amount_lent_to_B_is_5000 (loan : LoanDetails)
  (h1 : loan.rate = 12 / 100)
  (h2 : loan.timeB = 2)
  (h3 : loan.timeC = 4)
  (h4 : loan.amountC = 3000)
  (h5 : loan.totalInterest = 2640) :
  calculateAmountLentToB loan = 5000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_lent_to_B_is_5000_l290_29042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_with_gcd_seven_l290_29089

theorem count_integers_with_gcd_seven : 
  (Finset.filter (fun n : ℕ => 1 ≤ n ∧ n ≤ 150 ∧ Nat.gcd 21 n = 7) (Finset.range 151)).card = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_with_gcd_seven_l290_29089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_income_is_55000_l290_29008

/-- Represents the tax calculation method and the observed total tax --/
structure TaxSystem where
  q : ℚ  -- Base tax rate as a rational number
  taxRate1 : ℚ := q / 100  -- Tax rate for first $30000
  taxRate2 : ℚ := (q + 3) / 100  -- Tax rate for $30000 to $50000
  taxRate3 : ℚ := (q + 5) / 100  -- Tax rate for above $50000
  observedTaxRate : ℚ := (q + 45/100) / 100  -- Observed total tax rate

/-- Calculates the tax for a given income --/
def calculateTax (ts : TaxSystem) (income : ℚ) : ℚ :=
  ts.taxRate1 * 30000 +
  ts.taxRate2 * (min 50000 income - 30000) +
  ts.taxRate3 * max (income - 50000) 0

/-- Theorem stating that the annual income is $55000 --/
theorem annual_income_is_55000 (ts : TaxSystem) :
  ∃ (income : ℚ), income = 55000 ∧ calculateTax ts income = ts.observedTaxRate * income :=
sorry

#eval calculateTax { q := 5 } 55000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_income_is_55000_l290_29008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l290_29013

noncomputable def z : ℂ := Complex.I^2018 + 5 / (3 - 4 * Complex.I)

theorem imaginary_part_of_z : z.im = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l290_29013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_passes_through_independent_point_l290_29092

/-- The set of curves satisfying y' + a(x)y = b(x) -/
def S (a b : ℝ → ℝ) := {C : ℝ → ℝ | ∀ x, deriv C x + a x * C x = b x}

/-- The point P_k through which all tangent lines pass -/
noncomputable def P_k (k : ℝ) (a b : ℝ → ℝ) : ℝ × ℝ := (k + 1 / (a k), b k / (a k))

/-- The tangent line to a curve C at a point x -/
noncomputable def TangentLine (C : ℝ → ℝ) (x : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t, p = (x + t, C x + t * deriv C x)}

/-- The line through two points -/
def LineThrough (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t, r = (p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2))}

theorem tangent_passes_through_independent_point
  (a b : ℝ → ℝ) (ha : ∀ x, a x ≠ 0) (hb : ∀ x, b x ≠ 0) (k : ℝ) (C : ℝ → ℝ) (hC : C ∈ S a b) :
  TangentLine C k = LineThrough (k, C k) (P_k k a b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_passes_through_independent_point_l290_29092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_equation_min_sum_distances_equation_l290_29085

-- Define the line l
noncomputable def line_l (k : ℝ) : ℝ → ℝ := λ x ↦ k * x - 4 * k + 1

-- Define the point P
def P : ℝ × ℝ := (4, 1)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define point A (x-intercept)
noncomputable def A (k : ℝ) : ℝ × ℝ := ((4 * k - 1) / k, 0)

-- Define point B (y-intercept)
noncomputable def B (k : ℝ) : ℝ × ℝ := (0, 1 - 4 * k)

-- Define the area of triangle OAB
noncomputable def area_OAB (k : ℝ) : ℝ := (4 * k - 1) * (1 - 4 * k) / (-2 * k)

-- Define the sum of distances |OA| + |OB|
noncomputable def sum_distances (k : ℝ) : ℝ := (1 - 4 * k) / (-k) + (1 - 4 * k)

-- Theorem for minimum area
theorem min_area_equation :
  ∃ k : ℝ, k < 0 ∧ line_l k (P.1) = P.2 ∧
  (∀ k' : ℝ, k' < 0 ∧ line_l k' (P.1) = P.2 → area_OAB k ≤ area_OAB k') ∧
  (λ x ↦ -1/4 * x + 2) = line_l k := by sorry

-- Theorem for minimum sum of distances
theorem min_sum_distances_equation :
  ∃ k : ℝ, k < 0 ∧ line_l k (P.1) = P.2 ∧
  (∀ k' : ℝ, k' < 0 ∧ line_l k' (P.1) = P.2 → sum_distances k ≤ sum_distances k') ∧
  (λ x ↦ -1/2 * x + 3) = line_l k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_equation_min_sum_distances_equation_l290_29085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_implies_m_range_l290_29002

/-- The function f(x) = (x^2 + 4x + 3) / (mx^2 + 4mx + 3) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (x^2 + 4*x + 3) / (m*x^2 + 4*m*x + 3)

/-- The set of m values for which f has domain ℝ -/
def M : Set ℝ := {m : ℝ | ∀ x, m*x^2 + 4*m*x + 3 ≠ 0}

theorem function_domain_implies_m_range :
  M = Set.Ici 0 := by
  sorry

#check function_domain_implies_m_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_implies_m_range_l290_29002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l290_29059

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Calculate the distance from a point to a line -/
noncomputable def Point.distanceToLine (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- The line l passing through M(2, 2) and equidistant from A(2, 3) and B(6, -9) -/
noncomputable def l : Line := sorry

/-- Point M -/
def M : Point := ⟨2, 2⟩

/-- Point A -/
def A : Point := ⟨2, 3⟩

/-- Point B -/
def B : Point := ⟨6, -9⟩

theorem line_equation_proof :
  (l = ⟨5, 2, -14⟩ ∨ l = ⟨3, 1, -8⟩) ∧
  M.onLine l ∧
  A.distanceToLine l = B.distanceToLine l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l290_29059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_marble_count_l290_29005

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- The total number of marbles in the urn -/
def total_marbles (mc : MarbleCount) : ℕ :=
  mc.red + mc.white + mc.blue + mc.green + mc.yellow

/-- Checks if the given marble count satisfies the equal probability condition -/
def satisfies_condition (mc : MarbleCount) : Prop :=
  (Nat.choose mc.red 5 =
   Nat.choose mc.white 1 * Nat.choose mc.red 4) ∧
  (Nat.choose mc.red 5 =
   Nat.choose mc.white 1 * Nat.choose mc.blue 1 * Nat.choose mc.red 3) ∧
  (Nat.choose mc.red 5 =
   Nat.choose mc.white 1 * Nat.choose mc.blue 1 * Nat.choose mc.green 1 * Nat.choose mc.red 2) ∧
  (Nat.choose mc.red 5 =
   Nat.choose mc.white 1 * Nat.choose mc.blue 1 * Nat.choose mc.green 1 * Nat.choose mc.yellow 1 * Nat.choose mc.red 1)

/-- The theorem stating that the smallest number of marbles satisfying the condition is 41 -/
theorem smallest_marble_count : 
  ∃ (mc : MarbleCount), satisfies_condition mc ∧ 
  total_marbles mc = 41 ∧ 
  (∀ (mc' : MarbleCount), satisfies_condition mc' → total_marbles mc' ≥ 41) := by
  sorry

#check smallest_marble_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_marble_count_l290_29005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l290_29023

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 - 6*x + 8/3

theorem f_extrema :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-2-ε) (-2+ε), f x ≤ f (-2)) ∧
  f (-2) = 10 ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (3-ε) (3+ε), f x ≥ f 3) ∧
  f 3 = -32/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l290_29023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_g_monotonicity_0_to_1_g_monotonicity_1_to_1_over_a_g_monotonicity_a_greater_1_g_monotonicity_1_over_a_to_1_g_monotonicity_a_equals_1_l290_29046

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 * Real.log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - 2 * x

-- Theorem for the tangent line when a = 1
theorem tangent_line_at_one (x : ℝ) :
  (f 1 x) = x^2 - 2*x + 2 * Real.log x ∧
  (f 1 1) = -1 ∧
  (2 * x - 3 = 2 * (x - 1) + (f 1 1)) :=
by sorry

-- Theorems for the monotonicity of g(x)
theorem g_monotonicity_0_to_1 (a : ℝ) (x : ℝ) :
  0 < a ∧ a < 1 →
  (0 < x ∧ x < 1 ∨ 1/a < x) → Monotone (fun x => g a x) :=
by sorry

theorem g_monotonicity_1_to_1_over_a (a : ℝ) (x : ℝ) :
  0 < a ∧ a < 1 →
  1 < x ∧ x < 1/a → StrictAnti (fun x => g a x) :=
by sorry

theorem g_monotonicity_a_greater_1 (a : ℝ) (x : ℝ) :
  1 < a →
  (0 < x ∧ x < 1/a ∨ 1 < x) → Monotone (fun x => g a x) :=
by sorry

theorem g_monotonicity_1_over_a_to_1 (a : ℝ) (x : ℝ) :
  1 < a →
  1/a < x ∧ x < 1 → StrictAnti (fun x => g a x) :=
by sorry

theorem g_monotonicity_a_equals_1 (x : ℝ) :
  0 < x → Monotone (fun x => g 1 x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_g_monotonicity_0_to_1_g_monotonicity_1_to_1_over_a_g_monotonicity_a_greater_1_g_monotonicity_1_over_a_to_1_g_monotonicity_a_equals_1_l290_29046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_circle_l290_29019

-- Define the circle as a predicate on real numbers
def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 1 = 0

theorem max_value_on_circle :
  ∃ (M : ℝ), M = 1/2 ∧
  (∀ (a b : ℝ), is_on_circle a b → b/(a-3) ≤ M) ∧
  (∃ (a b : ℝ), is_on_circle a b ∧ b/(a-3) = M) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_circle_l290_29019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_answer_satisfies_equation_l290_29088

/-- The number of 30-seat buses in the original plan -/
def x : ℕ := sorry

/-- The number of 40-seat buses in the new plan -/
def a : ℕ := sorry

/-- The total number of seventh-grade students at the school -/
def answer : ℕ := sorry

/-- The equation representing the relationship between the two bus plans -/
axiom bus_equation : 40 * (a - 1) + 35 = 30 * (a + 1) + 15

/-- The original plan leaves 15 students without seats -/
axiom original_plan : answer = 30 * x + 15

/-- The new plan requires one less bus than the original plan -/
axiom new_plan : x = a + 1

/-- Theorem stating that the answer satisfies the bus equation -/
theorem answer_satisfies_equation : 
  ∃ (a : ℕ), 40 * (a - 1) + 35 = 30 * (a + 1) + 15 ∧ answer = 30 * (a + 1) + 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_answer_satisfies_equation_l290_29088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l290_29014

open Real

noncomputable def f (x : ℝ) := sin x + cos x

theorem problem_statement (a : ℝ) (h : (deriv (deriv f)) a = 3 * f a) :
  (sin a)^2 - 3 / ((cos a)^2 + 1) = -14/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l290_29014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l290_29044

theorem matrix_transformation (P Q : Matrix (Fin 3) (Fin 3) ℝ) : 
  P = ![![3, 0, 0], ![0, 0, 1], ![0, 1, 0]] →
  (∀ a b c d e f g h i : ℝ, 
    Q = ![![a, b, c], ![d, e, f], ![g, h, i]] →
    P * Q = ![![3*a, 3*b, 3*c], ![g, h, i], ![d, e, f]]) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l290_29044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equality_l290_29071

def c : ℂ := 3 - 2*Complex.I
def d : ℂ := 1 + 3*Complex.I

theorem complex_expression_equality : 3*c + 4*d = 13 + 6*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equality_l290_29071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoming_age_problem_l290_29076

theorem xiaoming_age_problem :
  ∃ (x : ℕ) (k l : ℕ), 
    k ≠ l ∧
    k > 0 ∧ l > 0 ∧
    Int.natAbs (k - l) * x ≤ 10 ∧
    (∃ (a b : ℕ), k * (x - 1) + 1 = a ∧ l * (x - 1) + 1 = b) ∧
    (∃ (c d : ℕ), k * x = c ∧ l * x = d) ∧
    (∃ (e f : ℕ), k * (x + 1) - 1 = e ∧ l * (x + 1) - 1 = f) ∧
    x = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoming_age_problem_l290_29076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_production_no_loss_max_profit_l290_29033

-- Define the sales revenue function
noncomputable def P (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then
    -0.4 * x^2 + 4.2 * x - 0.8
  else if x > 5 then
    14.7 - 9 / (x - 3)
  else
    0

-- Define the cost function (in 10,000 yuan)
noncomputable def cost (x : ℝ) : ℝ := 2 + x

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ := P x - cost x

-- Theorem for minimum production to avoid loss
theorem min_production_no_loss :
  ∀ x : ℝ, x ≥ 1 → profit x ≥ 0 ∧ ∀ y : ℝ, y < 1 → profit y < 0 := by sorry

-- Theorem for maximum profit
theorem max_profit :
  ∃ x : ℝ, x = 6 ∧ profit x = 3.7 ∧ ∀ y : ℝ, y ≠ x → profit y < profit x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_production_no_loss_max_profit_l290_29033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l290_29081

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x
noncomputable def h (x : ℝ) : ℝ := x + 4 / x - 8
def g (b : ℝ) (x : ℝ) : ℝ := -x - 2 * b

-- State the theorem
theorem function_properties :
  ∀ (a : ℝ), a > 0 →
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ Real.sqrt a → f a x₁ ≥ f a x₂) ∧
  (∀ x₁ x₂ : ℝ, Real.sqrt a ≤ x₁ ∧ x₁ < x₂ → f a x₁ ≤ f a x₂) ∧
  (∀ x₁ x₂ : ℝ, -Real.sqrt a ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < 0 → f a x₁ ≥ f a x₂) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ -Real.sqrt a → f a x₁ ≤ f a x₂) ∧
  (∀ b : ℝ, (∀ x₁ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 3 →
    ∃ x₂ : ℝ, 1 ≤ x₂ ∧ x₂ ≤ 3 ∧ g b x₂ = h x₁) →
    1/2 ≤ b ∧ b ≤ 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l290_29081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_l290_29017

theorem max_prime_factors (x y : ℕ) 
  (h_gcd : (Nat.gcd x y).factors.length = 9)
  (h_lcm : (Nat.lcm x y).factors.length = 36)
  (h_x_pos : x > 0)
  (h_y_pos : y > 0)
  (h_fewer : (Nat.factors x).length < (Nat.factors y).length) : 
  (Nat.factors x).length ≤ 22 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_l290_29017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_multiplication_problem_l290_29056

/-- Represents a polynomial of degree 2 -/
structure MyPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The result of A's calculation -/
def A_result : MyPolynomial := ⟨6, 11, -10⟩

/-- The result of B's calculation -/
def B_result : MyPolynomial := ⟨2, -9, 10⟩

/-- Calculates the product of two linear polynomials -/
def multiply_linear (a b c d : ℤ) : MyPolynomial :=
  ⟨a * c, a * d + b * c, b * d⟩

theorem polynomial_multiplication_problem :
  ∃ (a b : ℤ),
    (multiply_linear 2 (-a) 3 b = A_result) ∧
    (multiply_linear 2 a 1 b = B_result) ∧
    (a = -5 ∧ b = -2) ∧
    (multiply_linear 2 a 3 b = MyPolynomial.mk 6 (-19) 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_multiplication_problem_l290_29056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l290_29048

/-- The trajectory of point M given the specified conditions -/
theorem trajectory_of_M (x y : ℝ) : 
  let F : ℝ × ℝ := (5, 0)
  let distance_MF := Real.sqrt ((x - F.1)^2 + y^2)
  let distance_M_line := |x - 9/5|
  (distance_MF / distance_M_line = 5/3) → (x^2 / 9 - y^2 / 16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l290_29048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rest_time_is_five_minutes_l290_29011

/-- Represents the walking and resting scenario of a man -/
structure WalkingScenario where
  speed : ℝ
  totalDistance : ℝ
  totalTime : ℝ
  restFrequency : ℝ

/-- Calculates the rest time per stop given a walking scenario -/
noncomputable def restTimePerStop (scenario : WalkingScenario) : ℝ :=
  let walkingTime := scenario.totalDistance / scenario.speed * 60
  let totalRestTime := scenario.totalTime - walkingTime
  let numberOfStops := scenario.totalDistance / scenario.restFrequency - 1
  totalRestTime / numberOfStops

/-- Theorem stating that the rest time per stop is 5 minutes for the given scenario -/
theorem rest_time_is_five_minutes :
  let scenario : WalkingScenario := {
    speed := 10,
    totalDistance := 5,
    totalTime := 50,
    restFrequency := 1
  }
  restTimePerStop scenario = 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rest_time_is_five_minutes_l290_29011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jewelry_store_rings_l290_29070

theorem jewelry_store_rings (necklace_capacity : ℕ) (necklace_current : ℕ) 
  (ring_capacity : ℕ) (bracelet_capacity : ℕ) (bracelet_current : ℕ) 
  (necklace_price : ℕ) (ring_price : ℕ) (bracelet_price : ℕ) 
  (total_cost : ℕ) : ℕ :=
  by
  have h1 : necklace_capacity = 12 := by sorry
  have h2 : necklace_current = 5 := by sorry
  have h3 : ring_capacity = 30 := by sorry
  have h4 : bracelet_capacity = 15 := by sorry
  have h5 : bracelet_current = 8 := by sorry
  have h6 : necklace_price = 4 := by sorry
  have h7 : ring_price = 10 := by sorry
  have h8 : bracelet_price = 5 := by sorry
  have h9 : total_cost = 183 := by sorry

  let necklace_to_add := necklace_capacity - necklace_current
  let bracelet_to_add := bracelet_capacity - bracelet_current
  let necklace_cost := necklace_to_add * necklace_price
  let bracelet_cost := bracelet_to_add * bracelet_price
  let ring_cost := total_cost - necklace_cost - bracelet_cost
  let rings_to_add := ring_cost / ring_price
  let current_rings := ring_capacity - rings_to_add

  exact current_rings

#check jewelry_store_rings

example : jewelry_store_rings 12 5 30 15 8 4 10 5 183 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jewelry_store_rings_l290_29070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_h_roots_constraint_l290_29099

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (a b x : ℝ) : ℝ := (a / 2) * x + b
noncomputable def h (a b x : ℝ) : ℝ := f x * g a b x

-- Define the maximum value function
noncomputable def φ (a : ℝ) : ℝ :=
  if a < -2 then -a / 2 * Real.exp (-2 / a) else Real.exp 1

-- Part 1 theorem
theorem max_value_h (a : ℝ) :
  let b := 1 - a / 2
  (∀ x, x ∈ Set.Icc 0 1 → h a b x ≤ φ a) ∧
  (∃ x, x ∈ Set.Icc 0 1 ∧ h a b x = φ a) := by sorry

-- Part 2 theorem
theorem roots_constraint (b : ℝ) :
  (∃! x₁ x₂, x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ x₁ ≠ x₂ ∧ f x₁ = g 4 b x₁ ∧ f x₂ = g 4 b x₂) ↔
  b ∈ Set.Ioo (2 - 2 * Real.log 2) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_h_roots_constraint_l290_29099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangents_and_points_l290_29066

/-- Two circles in a 2D plane -/
structure TwoCircles where
  c1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 225}
  c2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 - 30 * p.1 + p.2^2 + 189 = 0}

/-- Common tangents of two circles -/
def common_tangents (circles : TwoCircles) : Set (ℝ × ℝ → Prop) :=
  { l | ∃ (a b c : ℝ), l = (λ p ↦ a * p.1 + b * p.2 = c) }

/-- Tangency points of common tangents -/
def tangency_points (circles : TwoCircles) : Set (ℝ × ℝ) :=
  { p | p ∈ circles.c1 ∪ circles.c2 ∧ 
        ∃ l ∈ common_tangents circles, l p }

/-- Main theorem: Common tangents and tangency points of the given circles -/
theorem circles_tangents_and_points (circles : TwoCircles) :
  (common_tangents circles = {λ p ↦ 3 * p.1 + 4 * p.2 = 75, λ p ↦ 3 * p.1 - 4 * p.2 = 75}) ∧
  (tangency_points circles = {(9, 12), (9, -12), (18 * 3/5, 4 * 4/5), (18 * 3/5, -4 * 4/5)}) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangents_and_points_l290_29066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_g_iff_a_geq_half_l290_29055

-- Define the constant e
noncomputable def e : ℝ := Real.exp 1

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a - Real.log x
noncomputable def g (x : ℝ) : ℝ := 1 / x - e / Real.exp x

-- State the theorem
theorem f_greater_g_iff_a_geq_half :
  ∀ a : ℝ, (∀ x : ℝ, x > 1 → f a x > g x) ↔ a ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_g_iff_a_geq_half_l290_29055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_paint_needed_for_conversion_l290_29079

/-- Represents the composition of a paint mixture -/
structure PaintMix where
  red : ℚ
  blue : ℚ
  yellow : ℚ

/-- Calculates the total amount of a specific color in a paint mixture -/
def total_color (mix : PaintMix) (volume : ℚ) (color : PaintMix → ℚ) : ℚ :=
  color mix * volume / (mix.red + mix.blue + mix.yellow)

/-- Theorem stating the amount of blue paint needed to convert fuchsia and emerald to mauve -/
theorem blue_paint_needed_for_conversion 
  (fuchsia : PaintMix) 
  (mauve : PaintMix) 
  (emerald : PaintMix) 
  (fuchsia_volume : ℚ) 
  (emerald_volume : ℚ) : 
  fuchsia.red = 5 → 
  fuchsia.blue = 3 → 
  fuchsia.yellow = 0 →
  mauve.red = 3 → 
  mauve.blue = 5 → 
  mauve.yellow = 0 →
  emerald.red = 0 → 
  emerald.blue = 2 → 
  emerald.yellow = 4 →
  fuchsia_volume = 24 →
  emerald_volume = 30 →
  (total_color mauve ((total_color fuchsia fuchsia_volume (·.red)) / mauve.red) (·.blue) -
   (total_color fuchsia fuchsia_volume (·.blue) + total_color emerald emerald_volume (·.blue))) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_paint_needed_for_conversion_l290_29079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_PRT_measure_l290_29003

noncomputable section

-- Define the segment PQ and its properties
def PQ : ℝ := 12

-- Define the midpoint R
def R : ℝ := PQ / 2

-- Define the midpoint S of QR
def S : ℝ := R / 2

-- Define the radii of the semi-circles
def radius_PQ : ℝ := PQ / 2
def radius_QR : ℝ := R / 2

-- Define the areas of the semi-circles
def area_PQ : ℝ := Real.pi * radius_PQ^2
def area_QR : ℝ := Real.pi * radius_QR^2

-- Define the total area of both semi-circles
def total_area : ℝ := area_PQ + area_QR

-- Define the angle PRT in radians
def angle_PRT_rad : ℝ := 2 * (total_area / 2) / area_PQ

-- Convert the angle to degrees
def angle_PRT_deg : ℝ := angle_PRT_rad * (180 / Real.pi)

end noncomputable section

-- Theorem statement
theorem angle_PRT_measure : 
  ∀ ε > 0, |angle_PRT_deg - 112.5| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_PRT_measure_l290_29003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l290_29080

theorem min_value_of_f (x : ℝ) : (4 : ℝ)^x + (4 : ℝ)^(-x) - (2 : ℝ)^(x+1) - (2 : ℝ)^(1-x) + 5 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l290_29080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_vertical_line_segment_l290_29063

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Function to create a line from a point and an angle -/
noncomputable def lineFromPointAndAngle (p : Point) (angle : ℝ) : Line :=
  { slope := Real.tan angle,
    intercept := p.y - Real.tan angle * p.x }

/-- Function to find the intersection of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : Point :=
  { x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope),
    y := l1.slope * ((l2.intercept - l1.intercept) / (l1.slope - l2.slope)) + l1.intercept }

/-- The main theorem -/
theorem intersection_forms_vertical_line_segment 
  (a b c d : Point)
  (h_a : a = ⟨0, 0⟩)
  (h_b : b = ⟨0, 4⟩)
  (h_c : c = ⟨6, 4⟩)
  (h_d : d = ⟨6, 0⟩)
  (line_a : Line := lineFromPointAndAngle a (π/4))
  (line_b : Line := lineFromPointAndAngle b (-π/4))
  (line_c : Line := lineFromPointAndAngle c (-π/4))
  (line_d : Line := lineFromPointAndAngle d (π/4))
  (int1 : Point := intersectionPoint line_a line_c)
  (int2 : Point := intersectionPoint line_b line_d) :
  int1.x = int2.x ∧ int1.y ≠ int2.y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_vertical_line_segment_l290_29063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_debt_to_ryan_l290_29068

noncomputable def total_amount : ℚ := 48
noncomputable def ryan_fraction : ℚ := 2/3
noncomputable def ryan_debt_to_leo : ℚ := 10
noncomputable def leo_final_amount : ℚ := 19

theorem leo_debt_to_ryan :
  let ryan_amount : ℚ := ryan_fraction * total_amount
  let leo_initial_amount : ℚ := total_amount - ryan_amount
  let leo_amount_after_ryan_pays : ℚ := leo_initial_amount + ryan_debt_to_leo
  leo_amount_after_ryan_pays - leo_final_amount = 7 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_debt_to_ryan_l290_29068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_onion_root_tip_tetrad_observation_false_l290_29012

-- Define the basic types and structures
structure Cell where
  type : String
  division : String

-- Define the conditions
def locust_oocyte : Cell := { type := "locust_oocyte", division := "meiosis" }

def peach_flower_stamen : Nat := 10  -- Arbitrary number for demonstration
def peach_flower_pistil : Nat := 5   -- Arbitrary number for demonstration

axiom peach_flower_stamen_more_numerous : peach_flower_stamen > peach_flower_pistil

def synapsis_occurs_in_meiosis : Prop := True

def onion_root_tip : Cell := { type := "onion_root_tip", division := "mitosis" }

-- Define the statement to be proven false
def onion_root_tip_tetrad_observation : Prop := False

-- Theorem statement
theorem onion_root_tip_tetrad_observation_false :
  ¬onion_root_tip_tetrad_observation :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_onion_root_tip_tetrad_observation_false_l290_29012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_arrangement_theorem_l290_29027

def number_of_rings : ℕ := 9
def number_of_fingers : ℕ := 4
def rings_per_arrangement : ℕ := 6

def number_of_arrangements : ℕ := 
  Nat.choose number_of_rings rings_per_arrangement * 
  Nat.factorial rings_per_arrangement * 
  Nat.choose (number_of_rings - 1) (number_of_fingers - 1)

theorem ring_arrangement_theorem :
  number_of_arrangements = 5080320 ∧
  (toString number_of_arrangements).take 3 = "508" := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_arrangement_theorem_l290_29027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_8_6_equivalence_l290_29006

/-- Represents a digit in a given base --/
def Digit (base : ℕ) := { d : ℕ // d < base }

/-- Converts a two-digit number in a given base to base 10 --/
def toBase10 (base : ℕ) (tens : Digit base) (ones : Digit base) : ℕ :=
  base * tens.val + ones.val

theorem base_8_6_equivalence :
  ∃ (A : Digit 8) (C : Digit 6),
    toBase10 8 A ⟨C.val, by sorry⟩ = toBase10 6 C ⟨A.val, by sorry⟩ ∧
    toBase10 8 A ⟨C.val, by sorry⟩ = 47 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_8_6_equivalence_l290_29006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_plus_minus_one_l290_29030

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + Real.sqrt (x^2 + 1))

-- Define what it means for f to be odd
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem odd_function_implies_a_plus_minus_one (a : ℝ) :
  is_odd_function (f a) → a = 1 ∨ a = -1 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_plus_minus_one_l290_29030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_wins_count_l290_29034

theorem mark_wins_count (games_with_mark : ℕ) (games_with_jill : ℕ) (jill_win_rate : ℚ) (jenny_total_wins : ℕ) :
  games_with_mark = 10 →
  games_with_jill = 2 * games_with_mark →
  jill_win_rate = 3/4 →
  jenny_total_wins = 14 →
  games_with_mark - jenny_total_wins + (jill_win_rate * ↑games_with_jill).floor = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_wins_count_l290_29034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_comp_f_equals_3_has_three_solutions_l290_29049

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x + 2 else 3*x - 6

-- Define the composite function f ∘ f
noncomputable def f_comp_f (x : ℝ) : ℝ := f (f x)

-- Theorem statement
theorem f_comp_f_equals_3_has_three_solutions :
  ∃ (a b c : ℝ), (∀ x : ℝ, f_comp_f x = 3 ↔ x = a ∨ x = b ∨ x = c) ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_comp_f_equals_3_has_three_solutions_l290_29049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l290_29032

-- Define the diamond operation as noncomputable
noncomputable def diamond (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

-- State the theorem
theorem diamond_calculation :
  diamond (diamond 7 24) (diamond (-24) (-7)) = 25 * Real.sqrt 2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l290_29032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_theorem_l290_29094

/-- Given two classes with their student counts and average scores, 
    calculate the overall average score --/
noncomputable def overall_average (students_a students_b : ℕ) (avg_a avg_b : ℝ) : ℝ :=
  ((students_a : ℝ) * avg_a + (students_b : ℝ) * avg_b) / ((students_a + students_b) : ℝ)

/-- Theorem stating that the overall average score for the given conditions is 99 --/
theorem class_average_theorem :
  overall_average 45 55 110 90 = 99 := by
  sorry

-- Remove #eval as it's not computable
-- #eval overall_average 45 55 110 90

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_theorem_l290_29094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_day_competition_l290_29010

/-- Represents a class with a number of girls and boys -/
structure MyClass where
  girls : Nat
  boys : Nat

/-- Represents a grade with two classes -/
structure Grade where
  class1 : MyClass
  class2 : MyClass

theorem field_day_competition (grade4 grade5 : Grade) 
  (h1 : grade4.class1 = { girls := 12, boys := 13 })
  (h2 : grade4.class2 = { girls := 15, boys := 11 })
  (h3 : grade5.class1 = { girls := 9, boys := 13 })
  (h4 : grade5.class2 = { girls := 10, boys := 11 }) :
  (grade4.class1.boys + grade4.class2.boys + grade5.class1.boys + grade5.class2.boys) -
  (grade4.class1.girls + grade4.class2.girls + grade5.class1.girls + grade5.class2.girls) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_day_competition_l290_29010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_find_a_l290_29022

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin (2 * x) - (1 / 3) * Real.sin (3 * x)

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.cos (2 * x) - Real.cos (3 * x)

-- Theorem statement
theorem extreme_value_condition (a : ℝ) :
  (f_derivative a (π / 3) = 0) → a = 1 := by
  sorry

-- Main theorem
theorem find_a :
  ∃ a : ℝ, (f_derivative a (π / 3) = 0) ∧ a = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_find_a_l290_29022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_l290_29073

/-- Given a boat with speed in still water and distances covered downstream and upstream in the same time, calculate the stream speed. -/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) 
  (h1 : boat_speed = 24)
  (h2 : downstream_distance = 64)
  (h3 : upstream_distance = 32)
  (h4 : downstream_distance / (boat_speed + x) = upstream_distance / (boat_speed - x)) :
  x = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_l290_29073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_monochromatic_rectangle_l290_29021

/-- A coloring of the integer lattice points on a plane using p colors. -/
def Coloring (p : ℕ) := ℤ × ℤ → Fin p

/-- A rectangle on the integer lattice. -/
structure Rectangle where
  x₁ : ℤ
  y₁ : ℤ
  x₂ : ℤ
  y₂ : ℤ
  h₁ : x₁ < x₂
  h₂ : y₁ < y₂

/-- Theorem: For any coloring of the integer lattice points on a plane using p colors,
    there exists a rectangle whose vertices are all the same color. -/
theorem existence_of_monochromatic_rectangle (p : ℕ) (c : Coloring p) :
  ∃ (r : Rectangle), 
    c (r.x₁, r.y₁) = c (r.x₂, r.y₁) ∧ 
    c (r.x₁, r.y₁) = c (r.x₁, r.y₂) ∧ 
    c (r.x₁, r.y₁) = c (r.x₂, r.y₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_monochromatic_rectangle_l290_29021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_frequency_is_five_l290_29072

/-- A collection of subsets satisfying the problem conditions -/
structure SubsetCollection where
  S : Finset ℕ
  subsets : Finset (Finset ℕ)
  h_S_card : S.card = 30
  h_subsets_card : subsets.card = 10
  h_subset_size : ∀ A ∈ subsets, A.card = 3
  h_subset_of_S : ∀ A ∈ subsets, A ⊆ S
  h_pairwise_intersect : ∀ A B, A ∈ subsets → B ∈ subsets → A ≠ B → (A ∩ B).Nonempty

/-- The number of subsets an element belongs to -/
def element_frequency (c : SubsetCollection) (i : ℕ) : ℕ :=
  (c.subsets.filter (λ A => i ∈ A)).card

/-- The maximum frequency of any element -/
def max_frequency (c : SubsetCollection) : ℕ :=
  Finset.sup c.S (λ i => element_frequency c i)

/-- The main theorem -/
theorem min_max_frequency_is_five :
  ∀ c : SubsetCollection, 5 ≤ max_frequency c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_frequency_is_five_l290_29072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_area_perimeter_relation_l290_29001

/-- For a regular n-gon with perimeter k_n inscribed in a circle of radius r, 
    and a regular 2n-gon with area t_2n inscribed in a circle of the same radius r,
    the ratio of the area of the 2n-gon to the square of the radius 
    is equal to half the ratio of the perimeter of the n-gon to the diameter. -/
theorem regular_polygon_area_perimeter_relation (n : ℕ) (r : ℝ) (k_n t_2n : ℝ) :
  k_n > 0 → r > 0 → t_2n > 0 →
  k_n = 2 * n * r * Real.sin (π / n) →
  t_2n = n * r^2 * Real.sin (π / n) →
  t_2n / r^2 = k_n / (2 * r) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_area_perimeter_relation_l290_29001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_form_l290_29062

/-- An ellipse with foci at (1, 1) and (1, 7) passing through (12, -1) has the standard form equation (x-h)²/a² + (y-k)²/b² = 1 with the given values of a, b, h, and k. -/
theorem ellipse_standard_form (x y : ℝ) : 
  let f₁ : ℝ × ℝ := (1, 1)
  let f₂ : ℝ × ℝ := (1, 7)
  let p : ℝ × ℝ := (12, -1)
  let a : ℝ := (5/2) * (Real.sqrt 5 + Real.sqrt 37)
  let b : ℝ := (1/2) * Real.sqrt (1014 + 50 * Real.sqrt 185)
  let h : ℝ := 1
  let k : ℝ := 4
  (((x - f₁.1)^2 + (y - f₁.2)^2).sqrt + ((x - f₂.1)^2 + (y - f₂.2)^2).sqrt 
    = ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2).sqrt + ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2).sqrt) →
  ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_form_l290_29062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_35_l290_29060

def a : ℕ → ℕ
  | 15 => 20
  | n+1 => 50 * a n + (n+1)
  | _ => 0

def is_first_multiple_of_35 (n : ℕ) : Prop :=
  n > 15 ∧ 
  a n % 35 = 0 ∧ 
  ∀ k, 15 < k ∧ k < n → a k % 35 ≠ 0

theorem least_multiple_of_35 : 
  is_first_multiple_of_35 26 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_35_l290_29060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_of_H_l290_29090

/-- Represents a rectangle with real-valued sides. -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle. -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Given two rectangles H and H₁, where:
    1. The area of H is 1
    2. The perimeter of H₁ is 50% less than the perimeter of H
    3. The area of H₁ is 50% more than the area of H
    This theorem states that the minimum perimeter of H is 4√6. -/
theorem min_perimeter_of_H (H H₁ : Rectangle) 
  (area_H : area H = 1)
  (perimeter_relation : perimeter H₁ = perimeter H / 2)
  (area_relation : area H₁ = 3/2 * area H) :
  ∃ (min_perimeter : ℝ), 
    (∀ (H' : Rectangle), area H' = 1 → perimeter H' ≥ min_perimeter) ∧
    min_perimeter = 4 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_of_H_l290_29090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l290_29036

/-- The speed of the train given the specified conditions -/
noncomputable def train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed : ℝ) : ℝ :=
  let man_speed_ms := man_speed * 1000 / 3600
  let relative_speed := train_length / crossing_time
  let train_speed_ms := relative_speed + man_speed_ms
  train_speed_ms * 3600 / 1000

/-- Theorem stating the speed of the train under given conditions -/
theorem train_speed_approx :
  let ε := 0.00001  -- Small error tolerance
  let calculated_speed := train_speed 700 41.9966402687785 3
  abs (calculated_speed - 63.00468) < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l290_29036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_trip_expenses_l290_29053

noncomputable def trip_expenses (initial_money ticket_cost transportation_cost days daily_meal_cost tax_rate : ℝ) : ℝ :=
  let hotel_cost := ticket_cost / 2
  let meal_cost := days * daily_meal_cost
  let taxable_expenses := hotel_cost + transportation_cost
  let tourist_tax := taxable_expenses * tax_rate
  let total_expenses := ticket_cost + hotel_cost + transportation_cost + meal_cost + tourist_tax
  initial_money - total_expenses

theorem maria_trip_expenses :
  trip_expenses 760 300 80 5 40 0.1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_trip_expenses_l290_29053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chuck_bicycle_trip_l290_29050

/-- The distance traveled one way on a round trip, given total time, outbound speed, and return speed -/
noncomputable def distance_one_way (total_time : ℝ) (outbound_speed : ℝ) (return_speed : ℝ) : ℝ :=
  (total_time * outbound_speed * return_speed) / (outbound_speed + return_speed)

/-- Theorem stating that for the given conditions, the distance traveled one way is approximately 28.80 miles -/
theorem chuck_bicycle_trip :
  let total_time : ℝ := 3
  let outbound_speed : ℝ := 16
  let return_speed : ℝ := 24
  let calculated_distance := distance_one_way total_time outbound_speed return_speed
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |calculated_distance - 28.80| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chuck_bicycle_trip_l290_29050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_plus_sec_l290_29018

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.tan x + (1 / Real.cos x)

-- State the theorem
theorem period_of_tan_plus_sec :
  ∃ (p : ℝ), p > 0 ∧ 
  (∀ (x : ℝ), f (x + p) = f x) ∧ 
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_plus_sec_l290_29018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_functions_have_smallest_period_pi_l290_29041

noncomputable section

open Real

-- Define the four functions
def f₁ (x : ℝ) : ℝ := tan (x/2 - π/3)
def f₂ (x : ℝ) : ℝ := |sin x|
def f₃ (x : ℝ) : ℝ := sin x * cos x
def f₄ (x : ℝ) : ℝ := cos x + sin x

-- Define a predicate for a function having a period of π
def hasPeriodPi (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + π) = f x

-- Define a predicate for a function having a smallest positive period of π
def hasSmallestPeriodPi (f : ℝ → ℝ) : Prop :=
  hasPeriodPi f ∧ ∀ T, 0 < T → T < π → ¬(∀ x, f (x + T) = f x)

-- Theorem statement
theorem exactly_two_functions_have_smallest_period_pi :
  (¬hasSmallestPeriodPi f₁) ∧
  (hasSmallestPeriodPi f₂) ∧
  (hasSmallestPeriodPi f₃) ∧
  (¬hasSmallestPeriodPi f₄) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_functions_have_smallest_period_pi_l290_29041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_sum_to_one_expected_value_nine_days_l290_29024

-- Define the probability of answering correctly
variable (p : ℝ) (hp : 0 < p ∧ p < 1)

-- Define the distribution of X
noncomputable def P_X (x : ℕ) : ℝ :=
  match x with
  | 2 => (1 - p)^2
  | 3 => 3*p^3 - 4*p^2 + 2*p
  | 4 => -3*p^3 + 3*p^2
  | _ => 0

-- Define the expected value of points in one day
noncomputable def E_Z : ℝ := 8 * (11/27) + 10 * (16/27)

-- Theorem statements
theorem distribution_sum_to_one : 
  P_X p 2 + P_X p 3 + P_X p 4 = 1 := by sorry

theorem expected_value_nine_days : 
  p = 2/3 → 9 * E_Z = 248/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_sum_to_one_expected_value_nine_days_l290_29024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_homes_l290_29058

theorem distance_between_homes (uphill_speed downhill_speed : ℝ)
  (time_vasya_to_petya time_petya_to_vasya : ℝ) 
  (h1 : uphill_speed = 3)
  (h2 : downhill_speed = 6)
  (h3 : time_vasya_to_petya = 2.5)
  (h4 : time_petya_to_vasya = 3.5) :
  ∃ (distance : ℝ), 
    distance = 12 ∧
    (distance / uphill_speed + distance / downhill_speed = time_vasya_to_petya) ∧
    (distance / downhill_speed + distance / uphill_speed = time_petya_to_vasya) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_homes_l290_29058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l290_29078

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.B = 150 * (Real.pi / 180) ∧  -- Convert degrees to radians
  t.a = Real.sqrt 3 * t.c ∧
  t.b = 2 * Real.sqrt 7

-- Define the area function
noncomputable def triangle_area (t : Triangle) : Real :=
  (1 / 2) * t.a * t.c * Real.sin t.B

-- Define the angle condition
def angle_condition (t : Triangle) : Prop :=
  Real.sin t.A + Real.sqrt 3 * Real.sin t.C = Real.sqrt 2 / 2

-- State the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : triangle_conditions t) 
  (h2 : angle_condition t) : 
  triangle_area t = Real.sqrt 3 ∧ t.C = 15 * (Real.pi / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l290_29078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_three_l290_29015

theorem greatest_power_of_three (n : ℕ) : n = 603 →
  (∃ k : ℕ, (15^n - 6^n + 3^n) = 3^k * (5^n - 2^n + 1) ∧
   ∀ m : ℕ, m > k → ¬(3^m ∣ (15^n - 6^n + 3^n))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_three_l290_29015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_m_range_l290_29082

def quadratic_function (a c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 - 2 * a * x + c

theorem quadratic_function_m_range
  (a c : ℝ)
  (h_decreasing : ∀ x ∈ Set.Icc 0 1, StrictMonoOn (quadratic_function a c) (Set.Icc 0 1))
  (m : ℝ)
  (h_m : quadratic_function a c m ≤ quadratic_function a c 0) :
  m ∈ Set.Icc 0 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_m_range_l290_29082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l290_29067

/-- Two circles are externally tangent -/
def externally_tangent (r₁ r₂ : ℝ) : Prop := sorry

/-- A circle is internally tangent to another circle -/
def internally_tangent (r₁ r₂ : ℝ) : Prop := sorry

/-- A chord is a common external tangent of two circles within a larger circle -/
def is_common_external_tangent (chord r₁ r₂ r₃ : ℝ) : Prop := sorry

/-- Given three circles with radii 4, 8, and 12, where the circles with radii 4 and 8 
    are externally tangent to each other and internally tangent to the circle with radius 12, 
    the square of the length of the chord of the circle with radius 12 that is a common 
    external tangent of the other two circles is equal to 3584/9. -/
theorem chord_length_squared (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) (h₃ : r₃ = 12)
  (h_ext_tangent : externally_tangent r₁ r₂)
  (h_int_tangent₁ : internally_tangent r₁ r₃)
  (h_int_tangent₂ : internally_tangent r₂ r₃)
  (chord : ℝ) (h_chord : is_common_external_tangent chord r₁ r₂ r₃) :
  chord^2 = 3584/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l290_29067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_functions_max_min_l290_29093

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The maximum value of a function on an interval -/
def HasMaxOn (f : ℝ → ℝ) (a : ℝ) (I : Set ℝ) : Prop :=
  ∃ x ∈ I, f x = a ∧ ∀ y ∈ I, f y ≤ a

/-- The minimum value of a function on an interval -/
def HasMinOn (f : ℝ → ℝ) (a : ℝ) (I : Set ℝ) : Prop :=
  ∃ x ∈ I, f x = a ∧ ∀ y ∈ I, a ≤ f y

theorem odd_functions_max_min (f g : ℝ → ℝ) (F : ℝ → ℝ) :
  IsOdd f → IsOdd g →
  (∀ x, F x = f x + g x + 2) →
  HasMaxOn F 8 { x : ℝ | x > 0 } →
  HasMinOn F (-4) { x : ℝ | x < 0 } :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_functions_max_min_l290_29093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_implies_b_equals_two_l290_29016

-- Define the hyperbola
def is_hyperbola (b : ℝ) : ℝ → ℝ → Prop :=
  λ x y => x^2 - (y^2 / b^2) = 1

-- Define the asymptote
def is_asymptote (b : ℝ) : ℝ → ℝ → Prop :=
  λ x y => y = 2 * x

-- Theorem statement
theorem hyperbola_asymptote_implies_b_equals_two (b : ℝ) 
  (h1 : b > 0) 
  (h2 : ∃ x y : ℝ, is_hyperbola b x y ∧ is_asymptote b x y) : 
  b = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_implies_b_equals_two_l290_29016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_largest_factorable_n_l290_29075

/-- 
A function that checks if a quadratic expression 3x^2 + nx + 24 
can be factored as a product of two linear factors with integer coefficients
-/
def is_factorable (n : ℤ) : Prop :=
  ∃ (a b : ℤ), ∀ (x : ℤ), 3 * x^2 + n * x + 24 = (3 * x + a) * (x + b)

/-- 
Theorem stating that 73 is the largest integer n for which 
3x^2 + nx + 24 can be factored as a product of two linear factors 
with integer coefficients
-/
theorem largest_factorable_n : 
  (is_factorable 73 ∧ ∀ m : ℤ, m > 73 → ¬(is_factorable m)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_largest_factorable_n_l290_29075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_pole_distance_l290_29051

theorem new_pole_distance 
  (initial_poles : ℕ) 
  (initial_distance : ℝ) 
  (new_poles : ℕ) 
  (h1 : initial_poles = 402) 
  (h2 : initial_distance = 20) 
  (h3 : new_poles = 202) : 
  (initial_distance * (initial_poles - 1 : ℝ)) / (new_poles - 1 : ℝ) = 40 := by
  sorry

#check new_pole_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_pole_distance_l290_29051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l290_29097

/-- A geometric sequence with first term a₁ and common ratio q -/
noncomputable def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ :=
  λ n => a₁ * q ^ (n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_product (a₁ : ℝ) :
  ∃ q : ℝ, 
    a₁ = -1/2 ∧ 
    (geometric_sum a₁ q 6) / (geometric_sum a₁ q 3) = 7/8 ∧
    (geometric_sequence a₁ q 2) * (geometric_sequence a₁ q 4) = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l290_29097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_number_seventh_row_l290_29096

/-- Represents the lattice structure described in the problem -/
structure MyLattice where
  row_length : ℕ
  common_difference : ℕ
  first_row_start : ℕ

/-- Calculates the first number of a given row in the lattice -/
def first_number_of_row (l : MyLattice) (row : ℕ) : ℕ :=
  l.first_row_start + l.row_length * l.common_difference * (row - 1)

/-- Calculates the nth number in a given row of the lattice -/
def nth_number_in_row (l : MyLattice) (row : ℕ) (n : ℕ) : ℕ :=
  first_number_of_row l row + (n - 1) * l.common_difference

/-- The main theorem to be proved -/
theorem fifth_number_seventh_row (l : MyLattice) 
  (h1 : l.row_length = 8)
  (h2 : l.common_difference = 2)
  (h3 : l.first_row_start = 2) :
  nth_number_in_row l 7 5 = 106 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_number_seventh_row_l290_29096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_first_and_last_l290_29047

def numbers : List Int := [-3, 0, 5, 7, 10, 13]

def is_valid_arrangement (arr : List Int) : Prop :=
  arr.length = 6 ∧
  arr.toFinset = numbers.toFinset ∧
  arr.maximum? ≠ some (arr[4]!) ∧
  arr.maximum? ≠ some (arr[5]!) ∧
  arr.minimum? ≠ some (arr[0]!) ∧
  arr.minimum? ≠ some (arr[1]!) ∧
  arr[0]! + arr[5]! > 10

theorem average_of_first_and_last (arr : List Int) 
  (h : is_valid_arrangement arr) : 
  (arr[0]! + arr[5]!) / 2 = 23 / 2 := by
  sorry

#eval (23 : Rat) / 2  -- This will evaluate to 11.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_first_and_last_l290_29047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_logarithmic_expression_equals_five_l290_29031

-- Define lg as the logarithm with base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem complex_logarithmic_expression_equals_five :
  (lg 5)^2 - (lg 2)^2 + 8^(2/3 : ℝ) * lg (Real.sqrt 2) - 0.6^(0 : ℝ) + 0.2^(-1 : ℝ) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_logarithmic_expression_equals_five_l290_29031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_forty_integers_l290_29045

/-- The arithmetic mean of forty successive positive integers beginning at 5 is 24.5 -/
theorem arithmetic_mean_of_forty_integers (start : ℕ) (count : ℕ) : start = 5 → count = 40 →
  (Finset.sum (Finset.range count) (λ i => start + i)) / count = (49 : ℚ) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_forty_integers_l290_29045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_formation_time_approx_l290_29039

/-- Calculates the time required for ice to reach a safe thickness -/
noncomputable def ice_formation_time (t_o : ℝ) (heat_release_rate : ℝ) (safe_thickness : ℝ) 
  (t_w : ℝ) (latent_heat : ℝ) (specific_heat : ℝ) (density : ℝ) : ℝ :=
  let mass := density * 1 * safe_thickness
  let energy_phase_change := latent_heat * mass
  let energy_cooling := specific_heat * mass * ((t_w - t_o) / 2)
  let total_energy := energy_phase_change + energy_cooling
  total_energy / heat_release_rate

/-- The time required for ice formation under given conditions is approximately 153.2 hours -/
theorem ice_formation_time_approx :
  let t_o := (-10 : ℝ)
  let heat_release_rate := (200 : ℝ)
  let safe_thickness := (0.1 : ℝ)
  let t_w := (0 : ℝ)
  let latent_heat := (330 : ℝ)
  let specific_heat := (2.1 : ℝ)
  let density := (900 : ℝ)
  abs (ice_formation_time t_o heat_release_rate safe_thickness t_w latent_heat specific_heat density - 153.2) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_formation_time_approx_l290_29039
