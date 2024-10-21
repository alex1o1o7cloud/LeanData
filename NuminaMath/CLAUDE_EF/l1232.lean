import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_way_split_l1232_123290

/-- Two positive real numbers are similar if their ratio is at most √2 --/
def similar (x y : ℝ) : Prop := 0 < x ∧ 0 < y ∧ x ≤ Real.sqrt 2 * y ∧ y ≤ Real.sqrt 2 * x

/-- A pile splitting is valid if all resulting piles are pairwise similar --/
def valid_split (piles : List ℝ) : Prop :=
  ∀ x y, x ∈ piles → y ∈ piles → similar x y

/-- The theorem stating that it's impossible to split a pile into three similar piles --/
theorem no_three_way_split (initial_pile : ℝ) (h : 0 < initial_pile) :
  ¬ ∃ (a b c : ℝ), a + b + c = initial_pile ∧ valid_split [a, b, c] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_way_split_l1232_123290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_value_l1232_123258

-- Define the polynomial
noncomputable def p (x : ℝ) : ℝ := x^3 - 3.9*x^2 + 4.4*x - 1.2

-- Define the triangle with its heights as roots of the polynomial
structure ScaleneTriangle where
  h : ℝ
  k : ℝ
  l : ℝ
  h_root : p h = 0
  k_root : p k = 0
  l_root : p l = 0
  scalene : h ≠ k ∧ k ≠ l ∧ h ≠ l

-- Define the inradius of the triangle
noncomputable def inradius (t : ScaleneTriangle) : ℝ :=
  1 / (1/t.h + 1/t.k + 1/t.l)

-- Theorem statement
theorem inradius_value (t : ScaleneTriangle) : inradius t = 3/11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_value_l1232_123258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_point_intersection_point_is_one_one_l1232_123274

-- Define the function g
def g (x : ℝ) : ℝ := x^3 - 3*x^2 + 9*x - 5

-- State the theorem
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, 
    let (c, d) := p
    g c = d ∧ g d = c ∧
    ∀ x y : ℝ, g x = y ∧ g y = x → (x, y) = (c, d) :=
by
  sorry

-- State that the unique intersection point is (1, 1)
theorem intersection_point_is_one_one :
  let p := Classical.choose unique_intersection_point
  p = (1, 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_point_intersection_point_is_one_one_l1232_123274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_distance_l1232_123214

/-- Represents the distance covered in miles -/
noncomputable def distance : ℝ := 1.5

/-- Represents the time interval in minutes -/
noncomputable def time_interval : ℝ := 8

/-- Represents the total riding time in minutes -/
noncomputable def total_time : ℝ := 40

/-- Calculates the total distance covered given a constant speed -/
noncomputable def total_distance (d : ℝ) (t : ℝ) (total : ℝ) : ℝ :=
  (d / t) * total

theorem bike_ride_distance :
  total_distance distance time_interval total_time = 7.5 := by
  -- Unfold the definitions
  unfold total_distance distance time_interval total_time
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_distance_l1232_123214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_jet_max_distance_water_jet_max_distance_value_l1232_123233

/-- The horizontal distance traveled by water jet from a hole in a container -/
noncomputable def water_jet_distance (a x : ℝ) : ℝ := 2 * Real.sqrt (x * (a - x))

/-- Theorem: The horizontal distance of the water jet is maximized when the hole is at half the water height -/
theorem water_jet_max_distance (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), x ∈ Set.Icc 0 a ∧ 
  ∀ (y : ℝ), y ∈ Set.Icc 0 a → water_jet_distance a x ≥ water_jet_distance a y :=
by
  sorry

/-- Corollary: The maximum horizontal distance is equal to the water height -/
theorem water_jet_max_distance_value (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), x ∈ Set.Icc 0 a ∧ 
  water_jet_distance a x = a ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 a → water_jet_distance a y ≤ a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_jet_max_distance_water_jet_max_distance_value_l1232_123233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_piles_theorem_l1232_123257

/-- Represents the state of the three piles of stones -/
structure PileState :=
  (pile1 : ℕ)
  (pile2 : ℕ)
  (pile3 : ℕ)

/-- The operation of moving stones between piles -/
def move (s : PileState) (i j k : Fin 3) : PileState :=
  match i.val, j.val, k.val with
  | 0, 1, 2 => ⟨s.pile1 - 1, s.pile2 - 1, s.pile3 + 2⟩
  | 0, 2, 1 => ⟨s.pile1 - 1, s.pile2 + 2, s.pile3 - 1⟩
  | 1, 0, 2 => ⟨s.pile1 - 1, s.pile2 - 1, s.pile3 + 2⟩
  | 1, 2, 0 => ⟨s.pile1 + 2, s.pile2 - 1, s.pile3 - 1⟩
  | 2, 0, 1 => ⟨s.pile1 - 1, s.pile2 + 2, s.pile3 - 1⟩
  | 2, 1, 0 => ⟨s.pile1 + 2, s.pile2 - 1, s.pile3 - 1⟩
  | _, _, _ => s  -- Default case to handle all other combinations

/-- The initial state of the piles -/
def initialState : PileState := ⟨19, 8, 9⟩

/-- The target state for the first question -/
def targetState1 : PileState := ⟨2, 12, 22⟩

/-- The target state for the second question -/
def targetState2 : PileState := ⟨12, 12, 12⟩

/-- Predicate to check if a state is reachable from the initial state -/
def isReachable (s : PileState) : Prop :=
  ∃ (n : ℕ) (moves : List (Fin 3 × Fin 3 × Fin 3)),
    moves.foldl (λ acc (m : Fin 3 × Fin 3 × Fin 3) => move acc m.1 m.2.1 m.2.2) initialState = s

theorem stone_piles_theorem :
  (isReachable targetState1) ∧ ¬(isReachable targetState2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_piles_theorem_l1232_123257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_problem_l1232_123253

theorem greatest_integer_problem (N : ℚ) : 
  (1 / (2 * 19).factorial + 1 / (3 * 18).factorial + 1 / (4 * 17).factorial + 
   1 / (5 * 16).factorial + 1 / (6 * 15).factorial + 1 / (7 * 14).factorial + 
   1 / (8 * 13).factorial + 1 / (9 * 12).factorial + 1 / (10 * 11).factorial = 
   N / (1 * 20).factorial) → 
  (⌊N / 100⌋ : ℤ) = 499 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_problem_l1232_123253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_escape_probability_l1232_123288

/-- Represents the probability of the frog escaping from a given pad -/
noncomputable def escape_prob : Fin 15 → ℝ := sorry

/-- The probability of jumping to the previous pad -/
noncomputable def jump_back_prob (n : Fin 15) : ℝ := (n.val ^ 2 : ℝ) / 196

/-- The probability of jumping to the next pad -/
noncomputable def jump_forward_prob (n : Fin 15) : ℝ := 1 - jump_back_prob n

/-- Represents whether a pad has a trap -/
def has_trap (n : Fin 15) : Bool := n.val = 7

theorem frog_escape_probability :
  escape_prob 0 = 0 ∧
  escape_prob 14 = 1 ∧
  (∀ n : Fin 15, 0 < n.val ∧ n.val < 14 →
    escape_prob n = 
      if has_trap n
      then (1/2) * (jump_back_prob n * escape_prob (n-1) + jump_forward_prob n * escape_prob (n+1))
      else jump_back_prob n * escape_prob (n-1) + jump_forward_prob n * escape_prob (n+1)) →
  escape_prob 2 = 108/343 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_escape_probability_l1232_123288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kolya_wins_l1232_123202

-- Define the game board
def board_width : ℕ := 5
def board_height : ℕ := 9

-- Define a position on the board
structure Position where
  x : ℕ
  y : ℕ

-- Define the initial and target positions
def initial_position : Position := ⟨1, 1⟩
def target_position : Position := ⟨board_width, board_height⟩

-- Define a valid move
def is_valid_move (start finish : Position) : Prop :=
  (start.x ≤ finish.x ∧ start.y = finish.y) ∨ (start.x = finish.x ∧ start.y ≤ finish.y)

-- Define Kolya's move condition
def is_kolya_move (pos : Position) : Prop :=
  (pos.x + pos.y) % 2 = 0

-- Theorem: Kolya wins the game
theorem kolya_wins :
  ∃ (winning_strategy : Position → Position),
    (∀ (pos : Position), is_valid_move pos (winning_strategy pos)) ∧
    (∀ (pos : Position), is_kolya_move pos → is_kolya_move (winning_strategy pos)) ∧
    (∃ (n : ℕ), (winning_strategy^[n] initial_position) = target_position) :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kolya_wins_l1232_123202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_and_line_l1232_123262

-- Define the curve and line functions
def curve (x : ℝ) : ℝ := 3 * x^2
def line (_ : ℝ) : ℝ := 3

-- Define the area function
noncomputable def enclosed_area : ℝ := ∫ x in (-1)..1, (line x - curve x)

-- Theorem statement
theorem area_enclosed_by_curve_and_line : enclosed_area = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_and_line_l1232_123262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_approx_l1232_123263

-- Define the segments of the journey
structure JourneySegment where
  speed : ℝ
  duration : ℝ

-- Define the journey
def journey : List JourneySegment := [
  ⟨90, 1⟩,
  ⟨75, 1⟩,
  ⟨110, 1⟩,
  ⟨65, 2⟩,
  ⟨95, 0.75⟩,
  ⟨80, 1.5⟩
]

-- Calculate total distance
noncomputable def totalDistance (j : List JourneySegment) : ℝ :=
  j.foldl (fun acc segment => acc + segment.speed * segment.duration) 0

-- Calculate total time
noncomputable def totalTime (j : List JourneySegment) : ℝ :=
  j.foldl (fun acc segment => acc + segment.duration) 0

-- Calculate average speed
noncomputable def averageSpeed (j : List JourneySegment) : ℝ :=
  totalDistance j / totalTime j

-- Theorem to prove
theorem journey_average_speed_approx :
  |averageSpeed journey - 82.24| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_approx_l1232_123263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_same_color_is_55_84_l1232_123281

def total_balls : ℕ := 9
def white_balls : ℕ := 2
def black_balls : ℕ := 3
def red_balls : ℕ := 4

noncomputable def probability_two_same_color : ℚ :=
  (Nat.choose white_balls 2 * Nat.choose (total_balls - white_balls) 1 +
   Nat.choose black_balls 2 * Nat.choose (total_balls - black_balls) 1 +
   Nat.choose red_balls 2 * Nat.choose (total_balls - red_balls) 1) /
  Nat.choose total_balls 3

theorem probability_two_same_color_is_55_84 :
  probability_two_same_color = 55 / 84 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_same_color_is_55_84_l1232_123281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_critical_point_l1232_123294

noncomputable def f (a : ℕ) (x : ℝ) : ℝ := Real.log x + a / (x + 1)

theorem unique_critical_point (a : ℕ) : 
  (∃! x : ℝ, x ∈ Set.Ioo 1 3 ∧ (deriv (f a)) x = 0) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_critical_point_l1232_123294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_graph_properties_l1232_123213

/-- Definition of a complete graph with n vertices -/
structure CompleteGraph (n : ℕ) where
  V : Type
  [fintype : Fintype V]
  card_eq : Fintype.card V = n

/-- Number of vertices in a complete graph -/
def numVertices {n : ℕ} (G : CompleteGraph n) : ℕ := n

/-- Number of edges in a complete graph -/
def numEdges {n : ℕ} (G : CompleteGraph n) : ℕ := n * (n - 1) / 2

/-- Theorem stating the number of vertices and edges for complete graphs K₁ to K₅ -/
theorem complete_graph_properties :
  (∀ (G : CompleteGraph 1), numVertices G = 1 ∧ numEdges G = 0) ∧
  (∀ (G : CompleteGraph 2), numVertices G = 2 ∧ numEdges G = 1) ∧
  (∀ (G : CompleteGraph 3), numVertices G = 3 ∧ numEdges G = 3) ∧
  (∀ (G : CompleteGraph 4), numVertices G = 4 ∧ numEdges G = 6) ∧
  (∀ (G : CompleteGraph 5), numVertices G = 5 ∧ numEdges G = 10) :=
by
  constructor
  · intro G; simp [numVertices, numEdges]
  constructor
  · intro G; simp [numVertices, numEdges]
  constructor
  · intro G; simp [numVertices, numEdges]
  constructor
  · intro G; simp [numVertices, numEdges]
  · intro G; simp [numVertices, numEdges]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_graph_properties_l1232_123213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrakaidecagon_area_approx_l1232_123218

/-- A tetrakaidecagon inscribed in a square -/
structure InscribedTetrakaidecagon where
  /-- The side length of the square -/
  square_side : ℝ
  /-- The number of segments each side of the square is divided into -/
  segments_per_side : ℕ
  /-- Assumption that the perimeter of the square is 56 meters -/
  h_perimeter : square_side * 4 = 56
  /-- Assumption that each side is divided into 7 segments -/
  h_segments : segments_per_side = 7

/-- The area of the inscribed tetrakaidecagon -/
noncomputable def tetrakaidecagon_area (t : InscribedTetrakaidecagon) : ℝ :=
  let square_area := t.square_side ^ 2
  let segment_length := t.square_side / t.segments_per_side
  let triangle_area := 1 / 2 * segment_length ^ 2
  let total_triangle_area := 16 * triangle_area
  square_area - total_triangle_area

/-- The main theorem stating the area of the tetrakaidecagon -/
theorem tetrakaidecagon_area_approx (t : InscribedTetrakaidecagon) :
  ∃ (ε : ℝ), ε > 0 ∧ |tetrakaidecagon_area t - 21.92| < ε := by
  sorry

#check tetrakaidecagon_area_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrakaidecagon_area_approx_l1232_123218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_plane_implies_perp_line_l1232_123282

-- Define the necessary structures
structure Line

structure Plane

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry

def perpendicular_lines (l1 l2 : Line) : Prop := sorry

def contained_in (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem line_perp_plane_implies_perp_line 
  (l m : Line) (a : Plane) 
  (h1 : perpendicular l a) 
  (h2 : contained_in m a) : 
  perpendicular_lines l m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_plane_implies_perp_line_l1232_123282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_between_sqrt16_and_sqrt128_l1232_123251

theorem whole_numbers_between_sqrt16_and_sqrt128 :
  (∀ n : ℕ, (16 : ℝ).sqrt < n ∧ n < (128 : ℝ).sqrt → n ∈ ({4, 5, 6, 7, 8, 9, 10, 11} : Finset ℕ)) ∧
  (∀ n : ℕ, n ∈ ({4, 5, 6, 7, 8, 9, 10, 11} : Finset ℕ) → (16 : ℝ).sqrt < n ∧ n < (128 : ℝ).sqrt) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_between_sqrt16_and_sqrt128_l1232_123251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_of_two_eq_three_sqrt_six_minus_thirteen_l1232_123267

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 3 * x - 4
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (3 * f x) - 3
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- State the theorem
theorem h_of_two_eq_three_sqrt_six_minus_thirteen : h 2 = 3 * Real.sqrt 6 - 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_of_two_eq_three_sqrt_six_minus_thirteen_l1232_123267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_1_statement_2_statement_3_statement_4_inverse_proportion_statements_l1232_123275

-- Define inverse proportion function
noncomputable def inverse_proportion (k : ℝ) : ℝ → ℝ := fun x => k / x

-- Statement 1
theorem statement_1 : Prop :=
  ∀ x : ℝ, x ≠ 0 → ∃ y : ℝ, inverse_proportion 2 x = y

-- Statement 2
theorem statement_2 : Prop :=
  ∃ f : ℝ → ℝ, ∃ k : ℝ, 
    (∀ x : ℝ, x ≠ 0 → f x = k / x) ∧
    (∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 → f (-x) = -f x ∧ f (1/x) = 1/(f x))

-- Statement 3
theorem statement_3 : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ < x₂ → inverse_proportion 3 x₁ > inverse_proportion 3 x₂

-- Statement 4
theorem statement_4 : Prop :=
  inverse_proportion (-6) 3 = -2 ∧ inverse_proportion (-6) (-3) = 2

-- Theorem stating that exactly three of the four statements are correct
theorem inverse_proportion_statements :
  (statement_1 ∧ statement_2 ∧ ¬statement_3 ∧ statement_4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_1_statement_2_statement_3_statement_4_inverse_proportion_statements_l1232_123275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l1232_123296

theorem triangle_cosine_inequality (A B C : ℝ) (h_triangle : A + B + C = π) :
  Real.cos (A / 2) + Real.cos (B / 2) + Real.cos (C / 2) ≥ 
  (Real.sqrt 3 / 2) * (Real.cos ((B - C) / 2) + Real.cos ((C - A) / 2) + Real.cos ((A - B) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l1232_123296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_l1232_123231

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_special_quadrilateral (q : Quadrilateral) : Prop :=
  let d_AB := Real.sqrt ((q.B.1 - q.A.1)^2 + (q.B.2 - q.A.2)^2)
  let d_BC := Real.sqrt ((q.C.1 - q.B.1)^2 + (q.C.2 - q.B.2)^2)
  let d_CD := Real.sqrt ((q.D.1 - q.C.1)^2 + (q.D.2 - q.C.2)^2)
  let d_DA := Real.sqrt ((q.A.1 - q.D.1)^2 + (q.A.2 - q.D.2)^2)
  let angle_ADC := Real.arccos ((d_CD^2 + d_DA^2 - (q.A.1 - q.C.1)^2 - (q.A.2 - q.C.2)^2) / (2 * d_CD * d_DA))
  d_AB = 10 ∧ d_BC = 10 ∧ d_CD = 17 ∧ d_DA = 17 ∧ angle_ADC = Real.pi/3

-- Theorem statement
theorem diagonal_length (q : Quadrilateral) (h : is_special_quadrilateral q) :
  Real.sqrt ((q.A.1 - q.C.1)^2 + (q.A.2 - q.C.2)^2) = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_l1232_123231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_A2B2C2_l1232_123259

structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

noncomputable def internal_bisector_point (T : Triangle) (Ω : Set (ℝ × ℝ)) : ℝ × ℝ × ℝ := sorry

noncomputable def smallest_angle (T : Triangle) : ℝ := min T.A (min T.B T.C)

theorem smallest_angle_A2B2C2 (T : Triangle) (Ω : Set (ℝ × ℝ)) :
  smallest_angle T = 40 →
  let A₁B₁C₁ := internal_bisector_point T Ω
  let T₁ : Triangle := {
    A := (T.B + T.C) / 2,
    B := (T.A + T.C) / 2,
    C := (T.A + T.B) / 2,
    sum_angles := sorry,
    positive_angles := sorry
  }
  let A₂B₂C₂ := internal_bisector_point T₁ Ω
  let T₂ : Triangle := {
    A := (T₁.B + T₁.C) / 2,
    B := (T₁.A + T₁.C) / 2,
    C := (T₁.A + T₁.B) / 2,
    sum_angles := sorry,
    positive_angles := sorry
  }
  smallest_angle T₂ = 65 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_A2B2C2_l1232_123259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_for_horizontal_asymptote_l1232_123240

noncomputable section

-- Define the numerator polynomial
def numerator (x : ℝ) : ℝ := 3 * x^7 + 4 * x^3 - 2 * x - 5

-- Define a general rational function with the given numerator
def rationalFunction (p : ℝ → ℝ) (x : ℝ) : ℝ := numerator x / p x

-- Define what it means for a function to have a horizontal asymptote
def hasHorizontalAsymptote (f : ℝ → ℝ) : Prop :=
  ∃ L : ℝ, ∀ ε > 0, ∃ M : ℝ, ∀ x > M, |f x - L| < ε

-- Define a notion of polynomial degree
def polyDegree (p : ℝ → ℝ) : ℕ := sorry

-- State the theorem
theorem smallest_degree_for_horizontal_asymptote :
  ∀ p : ℝ → ℝ, 
    (hasHorizontalAsymptote (rationalFunction p)) ∧ 
    (∀ q : ℝ → ℝ, hasHorizontalAsymptote (rationalFunction q) → polyDegree p ≤ polyDegree q) →
    polyDegree p = 7 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_for_horizontal_asymptote_l1232_123240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_one_third_optimal_one_station_optimal_two_stations_l1232_123230

-- Define the basic setup
structure TransportProblem where
  l : ℝ  -- Total distance between A and B
  a : ℝ  -- Capacity of one truck

-- Define efficiency calculation for one intermediate station
noncomputable def efficiency_one_station (p : TransportProblem) (d : ℝ) : ℝ :=
  (d * (p.l - d)) / (p.l ^ 2)

-- Define efficiency calculation for two intermediate stations
noncomputable def efficiency_two_stations (p : TransportProblem) (d1 d2 : ℝ) : ℝ :=
  ((d1 + d2) * (p.l - d1) * (p.l - d2)) / (p.l ^ 3)

-- Theorem for one intermediate station at 1/3 of the distance
theorem efficiency_one_third (p : TransportProblem) :
  efficiency_one_station p (p.l / 3) = 2 / 9 := by
  sorry

-- Theorem for optimal placement of one intermediate station
theorem optimal_one_station (p : TransportProblem) :
  ∀ d, 0 < d ∧ d < p.l → efficiency_one_station p d ≤ 1 / 4 := by
  sorry

-- Theorem for optimal placement of two intermediate stations
theorem optimal_two_stations (p : TransportProblem) :
  ∀ d1 d2, 0 < d1 ∧ 0 < d2 ∧ d1 + d2 < p.l →
    efficiency_two_stations p d1 d2 ≤ 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_one_third_optimal_one_station_optimal_two_stations_l1232_123230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_relationship_l1232_123215

-- Define the circle O
def Circle (O : EuclideanSpace ℝ (Fin 2)) (r : ℝ) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {P | ‖P - O‖ = r}

-- Define the line l
def Line (P Q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {R | ∃ t : ℝ, R = (1 - t) • P + t • Q}

-- Define tangency
def Tangent (s : Set (EuclideanSpace ℝ (Fin 2))) (c : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∃ p, p ∈ s ∩ c ∧ ∀ q, q ∈ s ∩ c → q = p

-- Define intersection
def Intersects (s t : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∃ p, p ∈ s ∩ t

-- State the theorem
theorem line_circle_relationship (O P : EuclideanSpace ℝ (Fin 2)) (l : Set (EuclideanSpace ℝ (Fin 2))) :
  (∃ r, r = 2 ∧ Circle O r = {X | ‖X - O‖ = r}) →
  (P ∈ l) →
  (‖P - O‖ = 2) →
  (Tangent l (Circle O 2) ∨ Intersects l (Circle O 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_relationship_l1232_123215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_matrix_det_zero_l1232_123228

open Real Matrix

noncomputable def cosine_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![Real.cos 0, Real.cos 1, Real.cos 2],
    ![Real.cos 3, Real.cos 4, Real.cos 5],
    ![Real.cos 6, Real.cos 7, Real.cos 8]]

theorem cosine_matrix_det_zero :
  Matrix.det cosine_matrix = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_matrix_det_zero_l1232_123228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_interval_l1232_123212

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2

theorem f_monotonic_increase_interval :
  ∀ (k : ℤ), ∃ (a b : ℝ),
    a = k * Real.pi - Real.pi / 3 ∧
    b = k * Real.pi + Real.pi / 6 ∧
    StrictMonoOn f (Set.Icc a b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_interval_l1232_123212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_range_l1232_123249

-- Define the function f(x) = log_a x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_base_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 > f a 3 → 0 < a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_range_l1232_123249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_a_is_315_l1232_123292

-- Define the reward function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (15 * x - a) / (x + 8)

-- Define the conditions
def isIncreasing (a : ℝ) : Prop :=
  ∀ x y, 50 ≤ x → x < y → y ≤ 500 → f a x < f a y

def isAboveMinimum (a : ℝ) : Prop :=
  ∀ x, 50 ≤ x → x ≤ 500 → f a x ≥ 7

def isBelowMaximum (a : ℝ) : Prop :=
  ∀ x, 50 ≤ x → x ≤ 500 → f a x ≤ 0.15 * x

-- The main theorem
theorem minimum_a_is_315 :
  ∃ (a : ℕ), a = 315 ∧
    isIncreasing (a : ℝ) ∧
    isAboveMinimum (a : ℝ) ∧
    isBelowMaximum (a : ℝ) ∧
    (∀ (b : ℕ), b < a →
      ¬(isIncreasing (b : ℝ) ∧ isAboveMinimum (b : ℝ) ∧ isBelowMaximum (b : ℝ))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_a_is_315_l1232_123292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_implies_sale_amount_l1232_123200

/-- Represents the commission structure for a sale -/
structure CommissionStructure where
  first_500_rate : Float
  excess_rate : Float

/-- Calculates the commission for a given sale amount and commission structure -/
def calculate_commission (amount : Float) (cs : CommissionStructure) : Float :=
  if amount <= 500 then
    amount * cs.first_500_rate
  else
    500 * cs.first_500_rate + (amount - 500) * cs.excess_rate

/-- Theorem stating that for the given commission structure, if the commission is approximately 31.25% of the sale amount, then the sale amount is approximately $800 -/
theorem commission_implies_sale_amount 
  (cs : CommissionStructure)
  (h1 : cs.first_500_rate = 0.20)
  (h2 : cs.excess_rate = 0.50)
  (sale_amount : Float)
  (h3 : calculate_commission sale_amount cs = 0.3125 * sale_amount) :
  sale_amount = 800 := by
  sorry

#check commission_implies_sale_amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_implies_sale_amount_l1232_123200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_l1232_123222

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (-x^2 + 2*x + 1)

-- State the theorem
theorem f_strictly_decreasing : 
  ∀ x y : ℝ, x < y ∧ y ≤ 1 → f y < f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_l1232_123222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_on_reciprocal_curve_l1232_123201

/-- The area of an equilateral triangle formed by two points on the curve y = 1/x in the first quadrant and the origin is √3. -/
theorem equilateral_triangle_area_on_reciprocal_curve : ∃ (a b : ℝ), 
  a > 0 ∧ b > 0 ∧  -- Points in the first quadrant
  b = 1 / a ∧      -- Points on the curve y = 1/x
  (a^2 + (1/a)^2 = b^2 + (1/b)^2) ∧  -- Equilateral triangle condition
  Real.sqrt 3 = (1/2) * ((a - b)^2 + ((1/a) - (1/b))^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_on_reciprocal_curve_l1232_123201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1232_123276

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Define the point on the curve
def point : ℝ × ℝ := (1, -1)

-- Define the slope of the tangent line at the point
def tangent_slope : ℝ := f' point.1

-- Define the equation of the tangent line
def tangent_line (x : ℝ) : ℝ := tangent_slope * (x - point.1) + point.2

-- Define the x-intercept of the tangent line
noncomputable def x_intercept : ℝ := 2 / 3

-- Define the y-intercept of the tangent line
def y_intercept : ℝ := tangent_line 0

-- Theorem: The area of the triangle is 2/3
theorem triangle_area : (1/2) * x_intercept * y_intercept = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1232_123276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_1_solution_inequality_2_solution_l1232_123236

-- Part 1
def inequality_1 (x : ℝ) : Prop := (2 : ℝ)^(x^2 - 3*x - 5) ≥ ((1/2) : ℝ)^(x + 2)

def solution_set_1 : Set ℝ := {x | x ≤ -1 ∨ x ≥ 3}

theorem inequality_1_solution : 
  {x : ℝ | inequality_1 x} = solution_set_1 := by sorry

-- Part 2
def inequality_2a (x : ℝ) : Prop := (2*x + 1) / (x - 3) > 1
def inequality_2b (x : ℝ) : Prop := x^2 + x - 20 ≤ 0

def solution_set_2 : Set ℝ := {x | (x ≥ -5 ∧ x < -4) ∨ (x > 3 ∧ x ≤ 4)}

theorem inequality_2_solution :
  {x : ℝ | inequality_2a x ∧ inequality_2b x} = solution_set_2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_1_solution_inequality_2_solution_l1232_123236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_sum_l1232_123242

/-- Parabola defined by y^2 = 4x with focus F -/
structure Parabola where
  F : ℝ × ℝ -- Focus of the parabola

/-- A point on the parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: Minimum value of |AF| + 4|BF| is 9 -/
theorem parabola_min_distance_sum (p : Parabola) (A B : ParabolaPoint p) 
    (h : ∃ (t : ℝ), (1 - t) • (A.x, A.y) + t • (B.x, B.y) = p.F) :
    distance (A.x, A.y) p.F + 4 * distance (B.x, B.y) p.F ≥ 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_sum_l1232_123242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_properties_l1232_123261

theorem sector_properties (α : Real) (r : Real) 
  (h1 : α = 120 * (Real.pi / 180))
  (h2 : r = 6) : 
  (α * r = 4 * Real.pi) ∧ ((1 / 2) * α * r^2 = 12 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_properties_l1232_123261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l1232_123225

theorem quadratic_inequality_solution_set :
  let S : Set ℝ := {x : ℝ | 2 * x^2 - 3 * x - 2 ≥ 0}
  S = Set.Ici 2 ∪ Set.Iic (-1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l1232_123225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_min_perimeter_l1232_123207

/-- A triangle with two sides along the x- and y-axes and the third side passing through (1, 2) -/
structure SpecialTriangle where
  /-- x-coordinate of the point on the x-axis -/
  b : ℝ
  /-- y-coordinate of the point on the y-axis -/
  a : ℝ
  /-- The third side passes through (1, 2) -/
  third_side_condition : 2 * b = a * (b + 1)

/-- The perimeter of the special triangle -/
noncomputable def perimeter (t : SpecialTriangle) : ℝ :=
  t.a + t.b + Real.sqrt (t.b^2 + t.a^2)

/-- The minimum perimeter of the special triangle -/
noncomputable def min_perimeter : ℝ := 3 + 2 * Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6

/-- Theorem stating that the minimum perimeter of the special triangle is 3 + 2√2 + √3 + √6 -/
theorem special_triangle_min_perimeter :
  ∃ (t : SpecialTriangle), ∀ (u : SpecialTriangle), perimeter t ≤ perimeter u ∧ perimeter t = min_perimeter := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_min_perimeter_l1232_123207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l1232_123277

theorem cos_theta_value (θ : Real) (h1 : 0 < θ) (h2 : θ < Real.pi) 
  (h3 : Real.sin (-Real.pi/3) = 1/3) :
  Real.cos θ = (-Real.sqrt 3 + 2 * Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l1232_123277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l1232_123223

/-- A circle defined by parametric equations -/
structure ParametricCircle where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The center of a circle -/
structure CircleCenter where
  h : ℝ
  k : ℝ

/-- Given circle defined by x = 2cos(θ) and y = 2sin(θ) + 2 -/
noncomputable def givenCircle : ParametricCircle where
  x := fun θ => 2 * Real.cos θ
  y := fun θ => 2 * Real.sin θ + 2

/-- Theorem: The center of the given circle is (0, 2) -/
theorem circle_center : 
  ∃ (c : CircleCenter), c.h = 0 ∧ c.k = 2 ∧
  ∀ (θ : ℝ), (givenCircle.x θ - c.h)^2 + (givenCircle.y θ - c.k)^2 = 2^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l1232_123223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_iff_a_equals_one_l1232_123271

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*(Real.log (x^2 + 2) / Real.log 2) + a^2 - 3

-- State the theorem
theorem unique_root_iff_a_equals_one :
  (∃! x, f 1 x = 0) ∧ ∀ a, (∃! x, f a x = 0) → a = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_iff_a_equals_one_l1232_123271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l1232_123208

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + a^(-x)

-- State the theorem
theorem sum_of_f_values (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 = 3) :
  f a 0 + f a 1 + f a 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l1232_123208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_exact_score_eq_l1232_123238

/-- A coin toss game where heads scores 1 point and tails scores 2 points -/
def CoinTossGame := Unit

/-- The probability of getting heads in a fair coin toss -/
noncomputable def probHeads : ℝ := 1/2

/-- The probability of getting tails in a fair coin toss -/
noncomputable def probTails : ℝ := 1/2

/-- The score for getting heads -/
def scoreHeads : ℕ := 1

/-- The score for getting tails -/
def scoreTails : ℕ := 2

/-- The probability of scoring exactly n points in the coin toss game -/
noncomputable def probExactScore (n : ℕ) : ℝ := 1/3 * (2 + (-1/2)^n)

/-- Theorem: The probability of scoring exactly n points in the coin toss game
    is equal to (1/3) * (2 + (-1/2)^n) -/
theorem prob_exact_score_eq (game : CoinTossGame) (n : ℕ) :
  probExactScore n = 1/3 * (2 + (-1/2)^n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_exact_score_eq_l1232_123238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_amount_l1232_123205

/-- Calculates simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Calculates compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

theorem compound_interest_amount : 
  let simple_principal : ℝ := 2625.0000000000027
  let simple_rate : ℝ := 0.08
  let compound_rate : ℝ := 0.10
  let time : ℝ := 2
  let compound_principal : ℝ := 4000
  simple_interest simple_principal simple_rate time = 
    (1/2) * compound_interest compound_principal compound_rate time →
  compound_principal = 4000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_amount_l1232_123205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_b_bounds_l1232_123287

noncomputable def a (n : ℕ) : ℝ := (n^2 + 1) / Real.sqrt (n^4 + 4)

noncomputable def b : ℕ → ℝ
  | 0 => 1
  | n + 1 => b n * a (n + 1)

theorem b_formula (n : ℕ) (h : n ≥ 1) :
  b n = Real.sqrt 2 * Real.sqrt (n^2 + 1) / Real.sqrt (n^2 + 2*n + 2) := by
  sorry

theorem b_bounds (n : ℕ) (h : n ≥ 1) :
  1 / ((n + 1)^3 : ℝ) < b n / Real.sqrt 2 - n / (n + 1) ∧
  b n / Real.sqrt 2 - n / (n + 1) < 1 / (n^3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_b_bounds_l1232_123287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_value_l1232_123270

def sequence_a : ℕ → ℚ
  | 0 => 1
  | n + 1 => sequence_a n + 1 / ((n + 2) * (n + 1))

theorem a_4_value : sequence_a 3 = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_value_l1232_123270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l1232_123224

/-- The sum of the infinite series ∑(1/(n(n+3))) for n from 1 to infinity -/
noncomputable def infiniteSeries : ℝ := ∑' n, 1 / (n * (n + 3))

/-- Theorem: The sum of the infinite series ∑(1/(n(n+3))) for n from 1 to infinity is equal to 11/18 -/
theorem infiniteSeriesSum : infiniteSeries = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l1232_123224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1232_123293

/-- Definition of an ellipse -/
def is_ellipse (a b : ℝ) (C : Set (ℝ × ℝ)) : Prop :=
  a > b ∧ b > 0 ∧ C = {(x, y) | x^2 / a^2 + y^2 / b^2 = 1}

/-- Definition of a circle -/
def is_circle (b : ℝ) (O : Set (ℝ × ℝ)) : Prop :=
  b > 0 ∧ O = {(x, y) | x^2 + y^2 = b^2}

/-- Left vertex of the ellipse -/
def left_vertex (a : ℝ) : ℝ × ℝ := (-a, 0)

/-- Left focus of the ellipse -/
noncomputable def left_focus (a b : ℝ) : ℝ × ℝ := (-Real.sqrt (a^2 - b^2), 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- Main theorem -/
theorem ellipse_eccentricity (a b : ℝ) (C O : Set (ℝ × ℝ)) :
  is_ellipse a b C →
  is_circle b O →
  (∃ P ∈ O, ∃ k : ℝ, ∀ P' ∈ O,
    distance P' (left_vertex a) / distance P' (left_focus a b) = k) →
  eccentricity a b = (Real.sqrt 5 - 1) / 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1232_123293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_geometric_sequence_b_max_l1232_123252

def sequence_a (n : ℕ+) : ℝ := sorry

-- Define the condition for the sequence a_n
axiom sequence_a_sum (n : ℕ+) : 
  (Finset.range n).sum (λ i => sequence_a ⟨i + 1, Nat.succ_pos i⟩) = n - sequence_a n

-- Define b_n
def sequence_b (n : ℕ+) : ℝ := (2 - n) * (sequence_a n - 1)

theorem sequence_a_geometric : 
  ∃ (r : ℝ), ∀ (n : ℕ+), sequence_a n - 1 = (-1/2) * (1/2)^(n.val - 1) := by sorry

theorem sequence_b_max : 
  ∃ (m : ℝ), m = 1/8 ∧ ∀ (n : ℕ+), sequence_b n ≤ m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_geometric_sequence_b_max_l1232_123252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1232_123299

noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.sin x - Real.cos x)

theorem function_properties (A : ℝ) (h1 : 0 < A) (h2 : A < π/4)
  (h3 : f (A/2) = 1 - 4*Real.sqrt 2/5) :
  (∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧
    (∀ q, q > 0 ∧ (∀ x, f (x + q) = f x) → p ≤ q)) ∧
  (∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x, f x = m)) ∧
  Real.cos A = 7 * Real.sqrt 2 / 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1232_123299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jelly_bean_theorem_l1232_123244

/-- Represents the types of jelly beans -/
inductive JellyBeanType
  | Grape
  | Vanilla
  | Strawberry

/-- Represents the number of jelly beans for each type -/
structure JellyBeanCount where
  grape : ℕ
  vanilla : ℕ
  strawberry : ℕ

/-- Represents the cost per jelly bean for each type -/
structure JellyBeanCost where
  grape : ℚ
  vanilla : ℚ
  strawberry : ℚ

/-- Calculates the total number of jelly beans -/
def totalJellyBeans (count : JellyBeanCount) : ℕ :=
  count.grape + count.vanilla + count.strawberry

/-- Calculates the total cost of all jelly beans -/
def totalCost (count : JellyBeanCount) (cost : JellyBeanCost) : ℚ :=
  cost.grape * count.grape + cost.vanilla * count.vanilla + cost.strawberry * count.strawberry

/-- Theorem stating the total number of jelly beans and their cost -/
theorem jelly_bean_theorem (count : JellyBeanCount) (cost : JellyBeanCost) :
  count.grape = 5 * count.vanilla + 50 →
  count.vanilla = 120 →
  3 * count.strawberry = 2 * count.vanilla →
  cost.grape = 8/100 →
  cost.vanilla = 5/100 →
  cost.strawberry = 7/100 →
  totalJellyBeans count = 850 ∧ totalCost count cost = 159/5 := by
  sorry

#eval (159 : ℚ) / 5  -- To verify that 159/5 is indeed equal to 63.80

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jelly_bean_theorem_l1232_123244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1232_123266

theorem problem_solution (t : ℝ) (p q r : ℕ) 
  (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 9/4)
  (h2 : (1 - Real.sin t) * (1 - Real.cos t) = p/q - Real.sqrt r)
  (h3 : 0 < p ∧ 0 < q ∧ 0 < r)
  (h4 : Nat.Coprime p q) :
  r + p + q = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1232_123266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_not_passing_through_point_l1232_123239

noncomputable def f (x : ℝ) : ℝ := 1 / x
noncomputable def g (x : ℝ) : ℝ := x^2
noncomputable def h (x : ℝ) : ℝ := -x + 1
noncomputable def k (x : ℝ) : ℝ := x^3

theorem function_not_passing_through_point :
  f 1 = 1 ∧ g 1 = 1 ∧ h 1 ≠ 1 ∧ k 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_not_passing_through_point_l1232_123239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_A_B_l1232_123206

def A : ℕ := (Finset.range 21).sum (fun i => (2*i+1) * (2*i+2)) + 43

def B : ℕ := 1 + (Finset.range 20).sum (fun i => (2*i+2) * (2*i+3)) + 42

theorem absolute_difference_A_B : |Int.ofNat A - Int.ofNat B| = 882 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_A_B_l1232_123206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_exists_l1232_123209

theorem no_such_function_exists : ¬∃ f : ℝ → ℝ, 
  (∀ x : ℝ, f (x^2) - f x^2 ≥ (1/4 : ℝ)) ∧ 
  Function.Injective f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_exists_l1232_123209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1232_123272

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (α + π/4) = Real.sqrt 5 / 5)
  (h2 : π/4 < α)
  (h3 : α < 3*π/4) : 
  Real.cos α = -Real.sqrt 10 / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1232_123272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_equals_interval_l1232_123220

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = -x^2 + 2*x - 1}
def B : Set ℝ := {y | ∃ x, y = 2*x + 1}

-- State the theorem
theorem A_intersect_B_equals_interval : A ∩ B = Set.Iic 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_equals_interval_l1232_123220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twice_underscored_count_l1232_123260

/-- Represents a table of numbers with underscoring rules -/
structure Table (m n : ℕ) where
  entries : Fin m → Fin n → ℝ
  k : ℕ
  l : ℕ
  k_le_m : k ≤ m
  l_le_n : l ≤ n
  column_underscored : ∀ j, ∃ S : Finset (Fin m), S.card = k ∧ ∀ i ∈ S, ∀ i' ∉ S, entries i j ≥ entries i' j
  row_underscored : ∀ i, ∃ S : Finset (Fin n), S.card = l ∧ ∀ j ∈ S, ∀ j' ∉ S, entries i j ≥ entries i j'

/-- The number of entries underscored twice is at least k * l -/
theorem twice_underscored_count (m n : ℕ) (t : Table m n) :
  ∃ S : Finset (Fin m × Fin n), S.card ≥ t.k * t.l ∧
  ∀ p ∈ S, (∃ S₁ : Finset (Fin m), S₁.card = t.k ∧ p.1 ∈ S₁ ∧ ∀ i' ∉ S₁, t.entries p.1 p.2 ≥ t.entries i' p.2) ∧
           (∃ S₂ : Finset (Fin n), S₂.card = t.l ∧ p.2 ∈ S₂ ∧ ∀ j' ∉ S₂, t.entries p.1 p.2 ≥ t.entries p.1 j') :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twice_underscored_count_l1232_123260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1232_123273

/-- An ellipse with the given properties has eccentricity √2 - 1 -/
theorem ellipse_eccentricity (a b c : ℝ) (h_ellipse : b^2 + c^2 = a^2) 
  (h_isosceles_right : b^2 / a = 2 * c) : 
  c / a = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1232_123273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_paired_with_fortyseven_l1232_123227

def numbers : List ℕ := [49, 29, 9, 40, 22, 15, 53, 33, 13, 47]

def isPairing (p : List (ℕ × ℕ)) : Prop :=
  p.length = 5 ∧ 
  (∀ x, x ∈ numbers → ∃ y, (x, y) ∈ p ∨ (y, x) ∈ p) ∧
  (∀ pair, pair ∈ p → pair.1 ∈ numbers ∧ pair.2 ∈ numbers)

def hasEqualSums (p : List (ℕ × ℕ)) : Prop :=
  ∃ s, ∀ pair ∈ p, pair.1 + pair.2 = s

theorem fifteen_paired_with_fortyseven :
  ∀ p : List (ℕ × ℕ), 
    isPairing p → hasEqualSums p → 
    (15, 47) ∈ p ∨ (47, 15) ∈ p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_paired_with_fortyseven_l1232_123227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l1232_123280

theorem right_triangle_properties (a b : ℝ) (h1 : a = 30) (h2 : b = 45) :
  let area := (1/2) * a * b
  let hypotenuse := Real.sqrt (a^2 + b^2)
  area = 675 ∧ hypotenuse = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l1232_123280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_price_correct_l1232_123245

noncomputable def gift_price (dave_money : ℚ) (kyle_less : ℚ) (snowboard_fraction : ℚ) 
               (lisa_more : ℚ) (discount : ℚ) : ℚ :=
  let kyle_initial := 3 * dave_money - kyle_less
  let kyle_after_snowboard := kyle_initial * (1 - snowboard_fraction)
  let lisa_money := kyle_after_snowboard + lisa_more
  let total_money := kyle_after_snowboard + lisa_money
  total_money / (1 - discount)

theorem gift_price_correct : 
  gift_price 46 12 (1/3) 20 (15/100) = 221.18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_price_correct_l1232_123245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_162_5_l1232_123210

/-- Represents the speed of the car in miles per hour. -/
def max_speed : ℝ := 65

/-- Represents the time taken for acceleration in hours. -/
def acceleration_time : ℝ := 2

/-- Represents the time taken for deceleration in hours. -/
def deceleration_time : ℝ := 3

/-- Calculates the distance covered during acceleration or deceleration. -/
noncomputable def distance_covered (speed : ℝ) (time : ℝ) : ℝ :=
  (speed / 2) * time

/-- Represents the total distance covered by the car during the trip. -/
noncomputable def total_distance : ℝ :=
  distance_covered max_speed acceleration_time + distance_covered max_speed deceleration_time

/-- Theorem stating that the total distance covered is 162.5 miles. -/
theorem total_distance_is_162_5 : total_distance = 162.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_162_5_l1232_123210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_divisible_sum_l1232_123278

theorem existence_of_divisible_sum (A B : Set ℕ) : 
  A.Nonempty → B.Nonempty → Disjoint A B → A ∪ B = Finset.range 10 →
  ∃ a b, a ∈ A ∧ b ∈ B ∧ 11 ∣ (a^3 + a*b^2 + b^3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_divisible_sum_l1232_123278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_less_cos_implies_obtuse_l1232_123234

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = Real.pi

-- Define what it means for a triangle to be obtuse
def is_obtuse (t : Triangle) : Prop :=
  t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2

-- State the theorem
theorem sin_product_less_cos_implies_obtuse (t : Triangle) :
  Real.sin t.A * Real.sin t.B < Real.cos t.C → is_obtuse t :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_less_cos_implies_obtuse_l1232_123234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_k_range_l1232_123221

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log x - k * x + 1

-- Theorem for part (1)
theorem monotonic_increase_interval (k : ℝ) (h : k = 2) :
  ∃ a b : ℝ, a = 0 ∧ b = 1/2 ∧
  ∀ x y, 0 < x ∧ x < y ∧ y < b → f k x < f k y :=
sorry

-- Theorem for part (2)
theorem k_range (k : ℝ) :
  (∀ x : ℝ, x > 0 → f k x ≤ 0) → k ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_k_range_l1232_123221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_small_area_l1232_123216

/-- A triangle represented by its vertices -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Calculate the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def pointInTriangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- A set of 5 points in a triangle -/
structure PointSet (t : Triangle) where
  points : Fin 5 → ℝ × ℝ
  inTriangle : ∀ i, pointInTriangle (points i) t

theorem three_points_small_area (t : Triangle) (ps : PointSet t) 
  (h : triangleArea t = 1) :
  ∃ i j k : Fin 5, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    triangleArea ⟨ps.points i, ps.points j, ps.points k⟩ ≤ 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_small_area_l1232_123216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l1232_123268

noncomputable def g (x : ℝ) : ℝ := Real.log (x + Real.sqrt (2 + x^2))

theorem g_neither_even_nor_odd :
  (∃ x : ℝ, g (-x) ≠ g x) ∧ (∃ x : ℝ, g (-x) ≠ -g x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l1232_123268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_through_origin_l1232_123286

-- Define the curve
noncomputable def curve (α : ℝ) (x : ℝ) : ℝ := x^α + 1

-- Define the derivative of the curve
noncomputable def curve_derivative (α : ℝ) (x : ℝ) : ℝ := α * x^(α - 1)

theorem tangent_through_origin (α : ℝ) :
  (curve α 1 = 2) →  -- The curve passes through (1, 2)
  (curve_derivative α 1 = (2 - 0) / (1 - 0)) →  -- The slope of the tangent line equals the slope between (0, 0) and (1, 2)
  α = 2 :=
by
  sorry

#check tangent_through_origin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_through_origin_l1232_123286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_triangle_area_l1232_123237

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
  t.A + t.B + t.C = Real.pi ∧
  t.c = t.b * (1 + 2 * Real.cos t.A)

-- Theorem 1
theorem angle_relation (t : Triangle) (h : triangle_conditions t) : t.A = 2 * t.B := by
  sorry

-- Theorem 2
theorem triangle_area (t : Triangle) (h : triangle_conditions t) 
  (ha : t.a = 3) (hb : t.B = Real.pi / 6) : 
  (1 / 2) * t.a * t.b = 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_triangle_area_l1232_123237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estate_value_l1232_123289

theorem estate_value (estate : ℝ) 
  (h1 : ∃ (daughter_share son1_share son2_share : ℝ),
    daughter_share + son1_share + son2_share = 3/4 * estate ∧
    daughter_share / son1_share = 5/3 ∧
    son1_share / son2_share = 3/2)
  (h2 : ∃ (gardener_share : ℝ), gardener_share = 400)
  (h3 : ∃ (daughter_share gardener_share : ℝ),
    daughter_share = 4 * gardener_share ∧ gardener_share = 400) :
  estate = 4266.67 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_estate_value_l1232_123289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x₀_l1232_123283

-- Define the circle
def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point M
def M (x₀ : ℝ) : ℝ × ℝ := (x₀, 2 - x₀)

-- Define the angle between two vectors
noncomputable def angle (v₁ v₂ : ℝ × ℝ) : ℝ := 
  Real.arccos ((v₁.1 * v₂.1 + v₁.2 * v₂.2) / (Real.sqrt (v₁.1^2 + v₁.2^2) * Real.sqrt (v₂.1^2 + v₂.2^2)))

-- Main theorem
theorem range_of_x₀ (x₀ : ℝ) : 
  (∃ N : ℝ × ℝ, is_on_circle N.1 N.2 ∧ angle (M x₀) N = π/6) → 0 ≤ x₀ ∧ x₀ ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x₀_l1232_123283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_theorem_l1232_123204

/-- Calculates the percentage of water in dried grapes -/
noncomputable def water_percentage_in_dried_grapes (fresh_water_percentage : ℝ) 
  (fresh_weight : ℝ) (dried_weight : ℝ) : ℝ :=
  let solid_content := (1 - fresh_water_percentage) * fresh_weight
  let water_in_dried := dried_weight - solid_content
  (water_in_dried / dried_weight) * 100

/-- Theorem stating that the percentage of water in dried grapes is 20% -/
theorem water_percentage_theorem :
  water_percentage_in_dried_grapes 0.8 40 10 = 20 := by
  -- Unfold the definition
  unfold water_percentage_in_dried_grapes
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_theorem_l1232_123204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_wins_odd_grid_bob_wins_even_grid_l1232_123255

/-- Represents a player in the grid game -/
inductive Player : Type
| Anna : Player
| Bob : Player

/-- Represents the game state -/
structure GameState where
  size : Nat
  grid : Array (Array Bool)
  currentPlayer : Player

/-- Counts the number of crosses in a 2x2 square starting at (i, j) -/
def CountCrossesIn2x2Square (state : GameState) (i j : Nat) : Nat :=
  sorry

/-- Places a cross at position (x, y) and updates the game state -/
def PlaceCross (state : GameState) (x y : Nat) : GameState :=
  sorry

/-- Defines a winning strategy for a player -/
def WinningStrategy (p : Player) (initialState : GameState) : Prop :=
  ∃ (strategy : GameState → Nat × Nat),
    ∀ (gameState : GameState),
      gameState.currentPlayer = p →
      (¬ ∃ (i j : Nat), CountCrossesIn2x2Square gameState i j = 3) →
      let (x, y) := strategy gameState
      ¬ ∃ (i j : Nat), CountCrossesIn2x2Square (PlaceCross gameState x y) i j = 3

/-- Theorem: Anna has a winning strategy on a 2013 × 2013 grid -/
theorem anna_wins_odd_grid :
  ∃ (initialState : GameState),
    initialState.size = 2013 ∧
    initialState.currentPlayer = Player.Anna ∧
    WinningStrategy Player.Anna initialState := by
  sorry

/-- Theorem: Bob has a winning strategy on a 2014 × 2014 grid -/
theorem bob_wins_even_grid :
  ∃ (initialState : GameState),
    initialState.size = 2014 ∧
    initialState.currentPlayer = Player.Anna ∧
    WinningStrategy Player.Bob initialState := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_wins_odd_grid_bob_wins_even_grid_l1232_123255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1232_123217

/-- Calculates the speed of a train given its length, platform length, and time to cross the platform. -/
noncomputable def trainSpeed (trainLength platformLength : ℝ) (timeToCross : ℝ) : ℝ :=
  let totalDistance := trainLength + platformLength
  let speedMps := totalDistance / timeToCross
  speedMps * 3.6

/-- Theorem stating that a train with given parameters has a specific speed. -/
theorem train_speed_calculation :
  let trainLength : ℝ := 1020
  let platformLength : ℝ := 396.78
  let timeToCross : ℝ := 50
  let calculatedSpeed := trainSpeed trainLength platformLength timeToCross
  ∃ ε > 0, |calculatedSpeed - 102.01| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1232_123217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l1232_123232

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 3

-- Define the distance function from a point to the line
noncomputable def dist_to_line (x y : ℝ) : ℝ := |x - y - 1| / Real.sqrt 2

-- Theorem statement
theorem max_distance_to_line :
  ∃ (x y : ℝ), circle_C x y ∧
  ∀ (x' y' : ℝ), circle_C x' y' →
  dist_to_line x y ≥ dist_to_line x' y' ∧
  dist_to_line x y = Real.sqrt 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l1232_123232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_equals_set_l1232_123297

-- Define the universal set U
def U : Set Int := {-2, -1, 0, 1, 2, 3}

-- Define set A
def A : Set Int := {-1, 0, 3}

-- Define set B
def B : Set Int := {1, 3}

-- Theorem statement
theorem complement_of_union_equals_set : 
  (U \ (A ∪ B)) = {-2, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_equals_set_l1232_123297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_and_q_truth_compound_prop_1_compound_prop_2_compound_prop_3_compound_prop_4_l1232_123226

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.sin x = Real.sqrt 5 / 2

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 + x + 1 > 0

-- Theorem stating the truth values of p and q
theorem p_and_q_truth : ¬p ∧ q := by sorry

-- Theorems for each compound proposition
theorem compound_prop_1 : ¬(p ∧ q) := by sorry
theorem compound_prop_2 : ¬(p ∧ ¬q) := by sorry
theorem compound_prop_3 : (¬p ∨ q) := by sorry
theorem compound_prop_4 : ¬(¬p ∨ ¬q) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_and_q_truth_compound_prop_1_compound_prop_2_compound_prop_3_compound_prop_4_l1232_123226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_between_15_and_16_l1232_123256

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  y = 2 * (floor x) + 3 ∧ y = 3 * (floor (x - 2)) + 5

-- State the theorem
theorem x_plus_y_between_15_and_16 (x y : ℝ) :
  system x y → ¬(∃ n : ℤ, x = n) → 15 < x + y ∧ x + y < 16 := by
  sorry

#check x_plus_y_between_15_and_16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_between_15_and_16_l1232_123256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_even_number_in_sequence_l1232_123247

theorem third_even_number_in_sequence : 
  ∀ (seq : List Nat),
    (∀ n ∈ seq, 2 ≤ n ∧ n ≤ 14 ∧ Even n) →
    seq.length = 7 →
    (seq.map (λ x => x^2)).sum = 560 →
    seq.get? 2 = some 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_even_number_in_sequence_l1232_123247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_vector_sum_l1232_123246

/-- Given a square ABCD with side length 1, prove that |AB→ + BC→ - CA→| = 2√2 -/
theorem square_vector_sum (A B C D : ℝ × ℝ) : 
  ‖(A.1 - B.1, A.2 - B.2)‖ = 1 → 
  ‖(B.1 - C.1, B.2 - C.2)‖ = 1 → 
  ‖(C.1 - D.1, C.2 - D.2)‖ = 1 → 
  ‖(D.1 - A.1, D.2 - A.2)‖ = 1 →
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 1 →
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 1 →
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = 1 →
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = 1 →
  ‖(B.1 - A.1, B.2 - A.2) + (C.1 - B.1, C.2 - B.2) - (A.1 - C.1, A.2 - C.2)‖ = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_vector_sum_l1232_123246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_feeding_ways_count_total_feeding_ways_l1232_123284

/-- Represents the number of animal pairs in the zoo -/
def num_pairs : Nat := 6

/-- Represents the total number of animals in the zoo -/
def total_animals : Nat := 2 * num_pairs

/-- Calculates the number of ways to feed the animals under the given conditions -/
def feeding_ways : Nat :=
  5 * 5 * 4 * 4 * 3 * 3 * 2 * 2 * 1 * 1

/-- Theorem stating that the number of ways to feed the animals is 14400 -/
theorem feeding_ways_count : feeding_ways = 14400 := by
  rfl

/-- Predicate representing that the feeding sequence starts with a male lion -/
def starts_with_male_lion : Prop := True

/-- Predicate representing that the feeding alternates between genders -/
def alternates_genders : Prop := True

/-- Predicate representing that a tiger cannot be fed immediately after a lion -/
def no_tiger_after_lion : Prop := True

/-- Main theorem proving that the number of ways to feed all animals under the given conditions is 14400 -/
theorem total_feeding_ways :
  feeding_ways = 14400 ∧
  starts_with_male_lion ∧
  alternates_genders ∧
  no_tiger_after_lion := by
  constructor
  · exact feeding_ways_count
  · constructor
    · trivial
    · constructor
      · trivial
      · trivial


end NUMINAMATH_CALUDE_ERRORFEEDBACK_feeding_ways_count_total_feeding_ways_l1232_123284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_217_l1232_123291

/-- Given an arithmetic sequence {a_n} where a_1 = 1 and a_2 = 3,
    prove that the 109th term is 217. -/
theorem arithmetic_sequence_217 :
  ∃ a : ℕ → ℤ,
    a 1 = 1 ∧
    a 2 = 3 ∧
    (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) ∧
    a 109 = 217 := by
  -- We'll construct the sequence explicitly
  let a (n : ℕ) : ℤ := 2 * n - 1
  
  -- Now we prove that this sequence satisfies all our conditions
  have h1 : a 1 = 1 := by simp [a]
  have h2 : a 2 = 3 := by simp [a]
  have h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1 := by
    intro n
    simp [a]
    ring
  have h_109 : a 109 = 217 := by simp [a]

  -- Finally, we use these to prove our theorem
  exact ⟨a, h1, h2, h_arith, h_109⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_217_l1232_123291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_all_equal_digits_l1232_123243

/-- A natural number is triangular if it can be expressed as n(n+1)/2 for some natural number n. -/
def IsTriangular (m : ℕ) : Prop :=
  ∃ n : ℕ, m = n * (n + 1) / 2

/-- A number has all digits equal to a if it can be expressed as a * (10^k - 1) / 9 for some k. -/
def AllDigitsEqualTo (m a : ℕ) : Prop :=
  ∃ k : ℕ, m = a * (10^k - 1) / 9

/-- The main theorem stating that the only values of a between 1 and 9 (inclusive)
    for which there exists a triangular number with all digits equal to a
    are 1, 3, 5, 6, and 8. -/
theorem triangular_all_equal_digits :
  ∀ a : ℕ, 1 ≤ a ∧ a ≤ 9 →
    (∃ m : ℕ, IsTriangular m ∧ AllDigitsEqualTo m a) ↔ a ∈ ({1, 3, 5, 6, 8} : Finset ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_all_equal_digits_l1232_123243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mall_comparison_l1232_123254

/-- Cost function for Mall A -/
noncomputable def cost_A (x : ℝ) : ℝ :=
  if x ≤ 1000 then x else 1000 + 0.9 * (x - 1000)

/-- Cost function for Mall B -/
noncomputable def cost_B (x : ℝ) : ℝ :=
  if x ≤ 500 then x else 500 + 0.95 * (x - 500)

theorem mall_comparison :
  (cost_B 850 < cost_A 850) ∧
  (∀ x > 1000, cost_A x = 100 + 0.9 * x ∧ cost_B x = 475 + 0.95 * x) ∧
  (cost_A 1700 < cost_B 1700) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mall_comparison_l1232_123254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l1232_123269

/-- Given a parabola y² = 2px containing the point (1, √5), 
    the distance from this point to the focus is 9/4 -/
theorem distance_to_focus (p : ℝ) : 
  (Real.sqrt 5)^2 = 2 * p * 1 → 
  (9 : ℝ) / 4 = p / 2 + 1 := by 
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l1232_123269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_product_l1232_123203

theorem sum_inverse_product : (12 : ℝ) * (1/3 + 1/4 + 1/6)⁻¹ = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_product_l1232_123203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_average_is_16_5_l1232_123250

/-- A function that generates all valid combinations of three single-digit and three double-digit numbers from digits 1 to 9 with no repetitions -/
def validCombinations : List (List Nat) := sorry

/-- A function that calculates the average of a list of numbers -/
def average (list : List Nat) : Rat :=
  (list.sum : Rat) / list.length

/-- The theorem stating that the smallest possible average is 16.5 -/
theorem smallest_average_is_16_5 :
  (validCombinations.map average).minimum? = some (33/2) := by sorry

#eval (33:Rat) / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_average_is_16_5_l1232_123250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l1232_123285

/-- Given vectors a and b, if (a - λb) ⊥ b, then λ = 3/5 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (lambda : ℝ) :
  a = (1, 3) →
  b = (3, 4) →
  (a.1 - lambda * b.1, a.2 - lambda * b.2) • b = 0 →
  lambda = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l1232_123285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_decreasing_f_l1232_123219

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + 2*x - 1) / x

def domain : Set ℝ := { x | x ≥ 3/7 }

theorem a_range_for_decreasing_f :
  ∀ a : ℝ, 
    (∀ x ∈ domain, ∀ y ∈ domain, x < y → f a x > f a y) →
    a ≤ -49/9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_decreasing_f_l1232_123219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_polygon_area_l1232_123279

/-- Represents a convex polygon in the sequence P_n -/
structure Polygon where
  n : ℕ
  area : ℝ

/-- Defines the sequence of polygons P_n -/
noncomputable def polygon_sequence : ℕ → Polygon
  | 0 => { n := 0, area := Real.sqrt 3 / 4 }
  | n + 1 => 
    let prev := polygon_sequence n
    { n := n + 1, area := prev.area - (1/3) * (Real.sqrt 3 / 4) * (2/9)^n }

/-- The theorem to be proved -/
theorem limit_of_polygon_area :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |((polygon_sequence n).area - Real.sqrt 3 / 7)| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_polygon_area_l1232_123279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_dimension_is_seven_l1232_123211

/-- The cost per square foot of insulation -/
def cost_per_sq_ft : ℝ := 20

/-- The total cost to insulate the tank -/
def total_cost : ℝ := 1640

/-- The length of the first dimension of the tank -/
def length : ℝ := 3

/-- The height of the tank -/
def tank_height : ℝ := 2

/-- Calculate the surface area of a rectangular tank -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- The second dimension of the tank satisfies the given conditions -/
theorem second_dimension_is_seven :
  ∃ w : ℝ, w = 7 ∧ surface_area length w tank_height * cost_per_sq_ft = total_cost := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_dimension_is_seven_l1232_123211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_tangent_lines_l1232_123235

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 7 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 10*y + 13 = 0

-- Define a tangent line to both circles
def is_tangent_to_both_circles (a b c : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    circle1 x1 y1 ∧ circle2 x2 y2 ∧
    a*x1 + b*y1 + c = 0 ∧ a*x2 + b*y2 + c = 0

-- Theorem statement
theorem exactly_three_tangent_lines :
  ∃! (tangent_lines : Finset (ℝ × ℝ × ℝ)),
    (∀ l ∈ tangent_lines, is_tangent_to_both_circles l.1 l.2.1 l.2.2) ∧
    (tangent_lines.card = 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_tangent_lines_l1232_123235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rose_blouse_price_l1232_123298

-- Define the discount percentage
def discount_percentage : ℚ := 18

-- Define the marked price
def marked_price : ℚ := 180

-- Define the function to calculate the discounted price
noncomputable def discounted_price (percentage : ℚ) (price : ℚ) : ℚ :=
  price * (1 - percentage / 100)

-- Theorem statement
theorem rose_blouse_price :
  discounted_price discount_percentage marked_price = 147.6 := by
  -- Unfold the definition of discounted_price
  unfold discounted_price
  -- Perform the calculation
  norm_num
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rose_blouse_price_l1232_123298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_force_calculation_l1232_123264

/-- Represents the force exerted by water on a trapezoidal dam. -/
noncomputable def water_force_on_dam (ρ g a b h : ℝ) : ℝ :=
  let pressure (x : ℝ) := ρ * g * x
  let width (x : ℝ) := b - x * (b - a) / h
  ∫ x in Set.Icc 0 h, pressure x * width x

/-- The theorem stating the force exerted by water on the given trapezoidal dam. -/
theorem water_force_calculation (ρ g a b h : ℝ) 
  (hρ : ρ = 1000)
  (hg : g = 10)
  (ha : a = 7.2)
  (hb : b = 12)
  (hh : h = 5) :
  water_force_on_dam ρ g a b h = 11000000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_force_calculation_l1232_123264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_floor_products_l1232_123295

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem sum_of_floor_products (x y : ℝ) : 
  x > 0 → y > 0 → (floor x * x = 36) → (floor y * y = 71) → x + y = 119/8 := by
  intro hx hy hx_eq hy_eq
  have x_eq : x = 6 := by
    sorry
  have y_eq : y = 71/8 := by
    sorry
  rw [x_eq, y_eq]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_floor_products_l1232_123295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_disks_for_profit_is_231_l1232_123265

/-- The minimum number of computer disks Maria must sell to make a profit of $150 -/
def min_disks_for_profit : ℕ :=
  let buy_price : ℚ := 8 / 5
  let sell_price : ℚ := 9 / 4
  let profit_per_disk : ℚ := sell_price - buy_price
  let target_profit : ℚ := 150
  (target_profit / profit_per_disk).ceil.toNat

theorem min_disks_for_profit_is_231 : min_disks_for_profit = 231 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_disks_for_profit_is_231_l1232_123265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1232_123248

theorem triangle_side_length (A B C : Real) (a b c : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi → 
  0 < a ∧ 0 < b ∧ 0 < c →
  a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C →
  a = 1 →
  b = 4 →
  c = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1232_123248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_convex_1978_gon_with_integer_angles_l1232_123241

/-- Represents a convex polygon with n sides and integer interior angles -/
structure ConvexPolygon (n : ℕ) where
  sides : n > 2
  angles : Fin n → ℕ
  convex : ∀ i : Fin n, angles i < 180
  sum_angles : (Finset.sum Finset.univ angles : ℕ) = (n - 2) * 180

/-- Theorem stating that no convex 1978-gon exists with all interior angles as integer degrees -/
theorem no_convex_1978_gon_with_integer_angles :
  ¬ ∃ (p : ConvexPolygon 1978), True := by
  sorry

#check no_convex_1978_gon_with_integer_angles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_convex_1978_gon_with_integer_angles_l1232_123241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_housewife_savings_percentage_l1232_123229

/-- The percentage saved when buying an item on sale -/
noncomputable def percentage_saved (sale_price : ℝ) (amount_saved : ℝ) : ℝ :=
  (amount_saved / (sale_price + amount_saved)) * 100

/-- Theorem stating that the percentage saved is approximately 12.12% -/
theorem housewife_savings_percentage : 
  let sale_price : ℝ := 29
  let amount_saved : ℝ := 4
  abs (percentage_saved sale_price amount_saved - 12.12) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_housewife_savings_percentage_l1232_123229
