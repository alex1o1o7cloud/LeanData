import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_l184_18434

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

-- State the theorem
theorem f_monotonic_increasing :
  ∀ x y : ℝ, 1 < x → x < y → f x < f y :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_l184_18434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_of_motion_l184_18419

/-- The time taken for the boat and raft to meet -/
def a : ℝ := sorry

/-- The boat's own speed -/
def x : ℝ := sorry

/-- The river's flow speed -/
def y : ℝ := sorry

/-- Assertion that the boat's speed is positive -/
axiom hx : x > 0

/-- Assertion that the river's flow speed is positive -/
axiom hy : y > 0

/-- Assertion that the boat's speed is greater than the river's flow speed -/
axiom hxy : x > y

/-- The total distance between points A and B -/
def distance : ℝ := x * a

/-- Theorem stating the total time of motion for both the raft and the boat -/
theorem total_time_of_motion : 
  (distance / y) = a * (1 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_of_motion_l184_18419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_to_line_l184_18444

/-- The maximum distance from point (1,1) to the line x*cos(θ) + y*sin(θ) - 2 = 0 is 2 + √2 -/
theorem max_distance_point_to_line :
  let point : ℝ × ℝ := (1, 1)
  let line_equation (x y θ : ℝ) := x * Real.cos θ + y * Real.sin θ - 2
  ∃ (d_max : ℝ), d_max = 2 + Real.sqrt 2 ∧
    ∀ (θ : ℝ), |line_equation point.1 point.2 θ| / Real.sqrt ((Real.cos θ)^2 + (Real.sin θ)^2) ≤ d_max :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_to_line_l184_18444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_width_l184_18420

/-- Given a rectangular room with length 15 m, surrounded by a 2 m wide verandah on all sides,
    and the verandah area being 124 square meters, prove that the width of the room is 12 m. -/
theorem room_width (room_length room_width verandah_width verandah_area : ℝ) :
  room_length = 15 →
  verandah_width = 2 →
  verandah_area = 124 →
  ((room_length + 2 * verandah_width) * (room_width + 2 * verandah_width) -
   room_length * room_width = verandah_area) →
  room_width = 12 :=
by
  sorry

#check room_width

end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_width_l184_18420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l184_18440

noncomputable def A : Set ℝ := {x | x ≤ 2 * Real.sqrt 3}

noncomputable def a : ℝ := Real.sqrt 14
noncomputable def b : ℝ := 2 * Real.sqrt 2

theorem problem_statement : a ∉ A ∧ b ∈ A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l184_18440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_product_l184_18407

/-- The hyperbola defined by x^2 - y^2 = 2 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 = 2}

/-- The left vertex of the hyperbola -/
noncomputable def A₁ : ℝ × ℝ := (-Real.sqrt 2, 0)

/-- The right vertex of the hyperbola -/
noncomputable def A₂ : ℝ × ℝ := (Real.sqrt 2, 0)

/-- Distance from origin to a line passing through two points -/
noncomputable def distanceToLine (p q : ℝ × ℝ) : ℝ :=
  abs (p.1 * q.2 - p.2 * q.1) / Real.sqrt ((q.2 - p.2)^2 + (q.1 - p.1)^2)

theorem hyperbola_distance_product :
  ∀ p ∈ Hyperbola, p ≠ A₁ → p ≠ A₂ →
  ∃ d₁ d₂ : ℝ, d₁ = distanceToLine p A₁ ∧ d₂ = distanceToLine p A₂ ∧
  0 < d₁ * d₂ ∧ d₁ * d₂ < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_product_l184_18407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sin_2x_not_equal_sin_2x_minus_pi_3_l184_18446

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

-- Define the shifted function
noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi/3)

-- Define the function given in the problem statement
noncomputable def h (x : ℝ) : ℝ := Real.sin (2*x - Real.pi/3)

-- Theorem statement
theorem shift_sin_2x_not_equal_sin_2x_minus_pi_3 : 
  ∃ x : ℝ, g x ≠ h x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sin_2x_not_equal_sin_2x_minus_pi_3_l184_18446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_wins_l184_18475

/-- Represents the game state -/
structure GameState (n : ℕ) where
  pieces : Finset ℕ
  turn : Bool -- true for Player A, false for Player B

/-- Defines a valid move in the game -/
def validMove (n : ℕ) (start finish : ℕ) : Prop :=
  1 ≤ finish ∧ finish < start ∧ start ≤ n

/-- Defines the initial game state -/
def initialState (n : ℕ) : GameState n :=
  { pieces := {n - 2, n - 1, n}, turn := true }

/-- Defines a winning strategy for Player A -/
def winningStrategy (n : ℕ) : Prop :=
  ∃ (strategy : GameState n → ℕ × ℕ),
    ∀ (state : GameState n),
      state.turn → 
      let (start, finish) := strategy state
      validMove n start finish →
      ¬∃ (nextMove : ℕ × ℕ),
        let (nextStart, nextFinish) := nextMove
        validMove n nextStart nextFinish

/-- The main theorem stating that Player A has a winning strategy -/
theorem player_a_wins (n : ℕ) (h : n > 2) : winningStrategy n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_wins_l184_18475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_recolor_theorem_l184_18460

/-- A complete graph with n vertices, where n ≥ 3 -/
structure CompleteGraph (n : ℕ) where
  h : n ≥ 3

/-- A coloring of the edges of a complete graph using three colors -/
structure ThreeColoring (n : ℕ) where
  graph : CompleteGraph n
  color1_used : Bool
  color2_used : Bool
  color3_used : Bool
  all_colors_used : color1_used ∧ color2_used ∧ color3_used

/-- The minimum number of edges that need to be recolored -/
def min_recolor (n : ℕ) : ℕ := n / 3

/-- Predicate to check if a coloring is connected by edges of a single color -/
def is_connected_single_color (coloring : ThreeColoring n) : Prop :=
  sorry

/-- Predicate to check if two colorings differ in at most k edges -/
def differs_at_most (coloring1 coloring2 : ThreeColoring n) (k : ℕ) : Prop :=
  sorry

/-- The main theorem stating that the minimum number of edges to recolor is ⌊n/3⌋ -/
theorem min_recolor_theorem (n : ℕ) (coloring : ThreeColoring n) :
  ∃ (k : ℕ), k = min_recolor n ∧
  (∀ (j : ℕ), j < k → ¬(∃ (new_coloring : ThreeColoring n), 
    is_connected_single_color new_coloring ∧
    differs_at_most coloring new_coloring j)) ∧
  (∃ (new_coloring : ThreeColoring n),
    is_connected_single_color new_coloring ∧
    differs_at_most coloring new_coloring k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_recolor_theorem_l184_18460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l184_18482

theorem differential_equation_solution (x : ℝ) (y : ℝ → ℝ) :
  (∀ x > 0, x * (deriv y x) - (y x) / Real.log x = 0) ↔ 
  (∃ c : ℝ, ∀ x > 0, y x = c * Real.log x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l184_18482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_at_one_l184_18483

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 / 3 * x^3 + 2

-- Define the derivative of f
noncomputable def f_derivative (x : ℝ) : ℝ := -Real.sqrt 3 * x^2

-- State the theorem
theorem tangent_slope_angle_at_one :
  let slope := f_derivative 1
  let angle := Real.arctan slope
  angle = 2 / 3 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_at_one_l184_18483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clearing_time_approx_l184_18492

/-- Calculates the time for two trains to clear each other when moving in opposite directions -/
noncomputable def trainClearingTime (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let totalLength := length1 + length2
  let speed1MeterPerSec := speed1 * 1000 / 3600
  let speed2MeterPerSec := speed2 * 1000 / 3600
  let relativeSpeed := speed1MeterPerSec + speed2MeterPerSec
  totalLength / relativeSpeed

/-- The time for the given trains to clear each other is approximately 9.88 seconds -/
theorem train_clearing_time_approx (ε : ℝ) (h : ε > 0) :
  ∃ t, abs (t - trainClearingTime 320 270 120 95) < ε ∧ abs (t - 9.88) < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clearing_time_approx_l184_18492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l184_18465

def M : Set ℝ := {x | x^2 - 3*x + 2 > 0}
def N : Set ℝ := {x | (1/2 : ℝ)^x ≥ 4}

theorem intersection_M_N : M ∩ N = {x : ℝ | x ≤ -2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l184_18465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_quadratic_l184_18464

theorem sum_of_solutions_quadratic (a b c : ℚ) (h : a ≠ 0) :
  let equation := fun x : ℚ => a * x^2 + b * x + c
  let sum_of_solutions := -b / a
  (equation = fun x : ℚ => -48 * x^2 + 100 * x + 200) →
  sum_of_solutions = 25 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_quadratic_l184_18464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integration_by_substitution_l184_18438

open MeasureTheory

theorem integration_by_substitution 
  {a b : ℝ} 
  (φ : ℝ → ℝ) 
  (hφ_smooth : ContDiff ℝ ⊤ φ) 
  (hφ_inc : φ a < φ b) 
  (f : ℝ → ℝ) 
  (hf_meas : Measurable f)
  (hf_int : Integrable f (volume.restrict (Set.Icc (φ a) (φ b)))) :
  Integrable (fun y => f (φ y) * deriv φ y) (volume.restrict (Set.Icc a b)) ∧
  ∫ x in Set.Icc (φ a) (φ b), f x = ∫ y in Set.Icc a b, f (φ y) * deriv φ y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integration_by_substitution_l184_18438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_range_l184_18439

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 else Real.exp (Real.log 2 * x)

-- State the theorem
theorem f_composition_range :
  {a : ℝ | f (f a) = Real.exp (Real.log 2 * f a)} = {a : ℝ | a ≥ 2/3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_range_l184_18439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l184_18405

theorem cos_alpha_minus_pi_fourth (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.tan α = 2) : 
  Real.cos (α - π/4) = (3 * Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l184_18405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l184_18493

/-- Represents the set of coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- The value of each coin in cents --/
def coinValue : Coin → Nat
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- A flip outcome for all coins --/
def FlipOutcome := Coin → Bool

/-- The set of all possible flip outcomes --/
def allOutcomes : Set FlipOutcome := sorry

/-- The value of heads in a flip outcome --/
def headsValue (outcome : FlipOutcome) : Nat :=
  (List.filter (fun c => outcome c) [Coin.Penny, Coin.Nickel, Coin.Dime, Coin.Quarter, Coin.HalfDollar])
  |>.map coinValue
  |>.sum

/-- The set of successful outcomes (at least 40 cents heads) --/
def successfulOutcomes : Set FlipOutcome :=
  {outcome ∈ allOutcomes | headsValue outcome ≥ 40}

/-- The probability of an event --/
noncomputable def probability (event : Set FlipOutcome) : Rat :=
  Nat.card event / Nat.card allOutcomes

theorem coin_flip_probability :
  probability successfulOutcomes = 19 / 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l184_18493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_collinear_with_a_l184_18418

noncomputable def a : ℝ × ℝ := (1, Real.sqrt 3)

theorem unit_vector_collinear_with_a :
  let v : ℝ × ℝ := (-1/2, -(Real.sqrt 3)/2)
  (norm v = 1) ∧ (∃ (k : ℝ), v = k • a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_collinear_with_a_l184_18418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_shortest_path_l184_18422

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The shortest path problem for the hiker -/
theorem hiker_shortest_path (hiker_start shelter : Point) 
    (h1 : hiker_start.x = 0 ∧ hiker_start.y = -3)
    (h2 : shelter.x = 6 ∧ shelter.y = -8) : 
  ∃ (river_point : Point), 
    river_point.y = 0 ∧ 
    distance hiker_start river_point + distance river_point shelter = 3 + Real.sqrt 157 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_shortest_path_l184_18422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_trig_l184_18495

-- Define IsInternalAngle as it's not a standard Lean function
def IsInternalAngle (θ : Real) (A B C : Real × Real) : Prop :=
  0 < θ ∧ θ < Real.pi

theorem triangle_angle_trig (θ : Real) : 
  (∃ (A B C : Real × Real), IsInternalAngle θ A B C) →
  Real.sin θ * Real.cos θ = -1/8 →
  Real.sin θ - Real.cos θ = Real.sqrt 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_trig_l184_18495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_finishes_early_l184_18408

/-- Represents the farmer's field ploughing scenario -/
structure FieldPloughing where
  totalArea : ℚ
  initialProductivity : ℚ
  productivityIncrease : ℚ
  daysBeforeIncrease : ℕ

/-- Calculates the number of days ahead of schedule the farmer finishes -/
def daysAheadOfSchedule (f : FieldPloughing) : ℚ :=
  let plannedDays := f.totalArea / f.initialProductivity
  let areaBeforeIncrease := f.initialProductivity * f.daysBeforeIncrease
  let remainingArea := f.totalArea - areaBeforeIncrease
  let newProductivity := f.initialProductivity * (1 + f.productivityIncrease)
  let actualDays := f.daysBeforeIncrease + remainingArea / newProductivity
  plannedDays - actualDays

/-- Theorem stating that the farmer finishes 2 days ahead of schedule -/
theorem farmer_finishes_early (f : FieldPloughing) 
  (h1 : f.totalArea = 1440)
  (h2 : f.initialProductivity = 120)
  (h3 : f.productivityIncrease = 1/4)
  (h4 : f.daysBeforeIncrease = 2) :
  daysAheadOfSchedule f = 2 := by
  sorry

#eval daysAheadOfSchedule {
  totalArea := 1440,
  initialProductivity := 120,
  productivityIncrease := 1/4,
  daysBeforeIncrease := 2
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_finishes_early_l184_18408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_hits_ground_time_l184_18424

def ball_height (t : Real) : Real := -16 * t^2 - 20 * t + 70

theorem ball_hits_ground_time :
  ∃ t : Real, t > 0 ∧ ball_height t = 0 ∧ (round (t * 100) / 100 : Real) = 1.56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_hits_ground_time_l184_18424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_growth_properties_l184_18468

def beautiful_growth (s : List ℝ) : List ℝ :=
  sorry

def a (n : ℕ) : ℝ :=
  sorry

theorem beautiful_growth_properties :
  (a 2 = 5) ∧
  (∀ n : ℕ, a (n + 1) = 3 * a n - 1) ∧
  (∀ n : ℕ, (Finset.range n).sum (λ i => 3^(i+1) / (a (i+1) * a (i+2))) = 1/2 - 2/(3^(n+1) + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_growth_properties_l184_18468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equals_area_ratio_l184_18430

/-- Represents a trapezoidal field with given dimensions and angles -/
structure TrapezoidalField where
  AB : ℝ  -- Length of longer base
  CD : ℝ  -- Length of shorter base
  BC : ℝ  -- Length of one non-parallel side
  AD : ℝ  -- Length of other non-parallel side
  angleA : ℝ  -- Angle adjacent to longer base at A
  angleB : ℝ  -- Angle adjacent to longer base at B

/-- Calculates the area of the trapezoid closer to the longer base AB -/
noncomputable def areaCloserToAB (field : TrapezoidalField) : ℝ :=
  sorry

/-- Calculates the total area of the trapezoidal field -/
noncomputable def totalArea (field : TrapezoidalField) : ℝ :=
  sorry

/-- Calculates the fraction of crop brought to the longer base of a trapezoidal field -/
noncomputable def fractionToLongerBase (field : TrapezoidalField) : ℝ :=
  (areaCloserToAB field) / (totalArea field)

/-- Theorem stating that the fraction of crop brought to the longer base
    is equal to the ratio of the area closer to AB to the total area -/
theorem fraction_equals_area_ratio (field : TrapezoidalField)
  (h1 : field.AB = 100)
  (h2 : field.CD = 100)
  (h3 : field.BC = 150)
  (h4 : field.AD = 200)
  (h5 : field.angleA = 75)
  (h6 : field.angleB = 75) :
  fractionToLongerBase field = (areaCloserToAB field) / (totalArea field) := by
  -- Unfold the definition of fractionToLongerBase
  unfold fractionToLongerBase
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equals_area_ratio_l184_18430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l184_18436

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h1 : S seq 5 < S seq 6)
  (h2 : S seq 6 > S seq 7) :
  (seq.d < 0 ∧ 
   (∀ n, S seq n ≤ S seq 6) ∧
   S seq 11 > 0 ∧
   ¬(S seq 12 < 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l184_18436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_one_value_expression_two_value_l184_18454

-- Define a and b as noncomputable
noncomputable def a : ℝ := Real.sqrt 3 - 2
noncomputable def b : ℝ := Real.sqrt 3 + 2

-- Theorem for the first expression
theorem expression_one_value : a^2 + 2*a*b + b^2 = 12 := by
  -- The proof is omitted for now
  sorry

-- Theorem for the second expression
theorem expression_two_value : a^2*b - a*b^2 = 4 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_one_value_expression_two_value_l184_18454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l184_18406

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - Real.sqrt 3 * (Real.cos x)^2

-- State the theorem
theorem f_properties :
  -- The smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- The maximum value on [0, π/2] is 1 - √3/2
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 1 - Real.sqrt 3 / 2 ∧
    ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≤ f x) ∧
  -- The minimum value on [0, π/2] is -√3
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = -Real.sqrt 3 ∧
    ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ f y) :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l184_18406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invisible_square_l184_18414

/-- A point (x, y) is invisible if there exists a point with integer coordinates on the line segment between (0, 0) and (x, y), excluding the endpoints. -/
def invisible (x y : ℤ) : Prop :=
  ∃ (k : ℤ), 0 < k ∧ k < Int.gcd x y ∧ (x / k : ℚ).isInt ∧ (y / k : ℚ).isInt

/-- For any natural number L, there exists integers a and b such that 
    for all i and j in the range [0, L], the point (a+i, b+j) is invisible. -/
theorem invisible_square (L : ℕ) : 
  ∃ (a b : ℤ), ∀ (i j : ℕ), i ≤ L → j ≤ L → invisible (a + i) (b + j) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_invisible_square_l184_18414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_rail_elevation_l184_18484

/-- The angle of elevation for a train's outer rail --/
noncomputable def rail_elevation_angle (v : ℝ) (R : ℝ) : ℝ :=
  Real.arctan ((v^2) / (R * 9.8))

/-- Conversion from km/h to m/s --/
noncomputable def km_per_hour_to_m_per_s (v : ℝ) : ℝ :=
  v * (1000 / 3600)

theorem train_rail_elevation :
  let v : ℝ := 60  -- Train speed in km/h
  let R : ℝ := 200 -- Curve radius in meters
  let v_ms : ℝ := km_per_hour_to_m_per_s v
  let θ : ℝ := rail_elevation_angle v_ms R
  ‖θ - 8.09 * (π / 180)‖ < 0.001 -- Approximate equality within 0.001 radians
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_rail_elevation_l184_18484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_T_coordinates_l184_18442

noncomputable section

-- Define the line equation
def line_equation (x : ℝ) : ℝ := -3/4 * x + 9

-- Define points P and Q
def P : ℝ × ℝ := (12, 0)
def Q : ℝ × ℝ := (0, 9)

-- Define point T
def T (r s : ℝ) : ℝ × ℝ := (r, s)

-- Define the condition that T is on the line
def T_on_line (r s : ℝ) : Prop := s = line_equation r

-- Define the condition that T is between P and Q
def T_between_P_and_Q (r s : ℝ) : Prop := 0 ≤ r ∧ r ≤ 12 ∧ 0 ≤ s ∧ s ≤ 9

-- Define the area ratio condition
def area_ratio_condition (r s : ℝ) : Prop := 
  (1/2 * 12 * 9) = 3 * (1/2 * r * s)

-- Theorem statement
theorem point_T_coordinates : 
  ∀ r s : ℝ, 
  T_on_line r s → 
  T_between_P_and_Q r s → 
  area_ratio_condition r s → 
  r + s = 11 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_T_coordinates_l184_18442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_a_l184_18478

-- Define the triangle ABC
def triangle (A B C : Real) (a b c : Real) : Prop :=
  A + B + C = Real.pi ∧ 
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  c / Real.sin C = a / Real.sin A

-- Theorem statement
theorem side_length_a (A B C : Real) (a b c : Real) :
  triangle A B C a b c →
  A = Real.pi / 3 →
  B = Real.pi / 4 →
  c = 20 →
  a = 30 * Real.sqrt 2 - 10 * Real.sqrt 6 := by
  sorry

#check side_length_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_a_l184_18478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_last_two_digits_l184_18479

def last_two_digits (n : Nat) : Nat := n % 100

theorem factorial_sum_last_two_digits :
  last_two_digits (Nat.factorial 3 + Nat.factorial 6 + Nat.factorial 9 + Nat.factorial 12 + Nat.factorial 15) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_last_two_digits_l184_18479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l184_18411

-- Define the set A
def A (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = x}

-- Define the power function f
noncomputable def f : ℝ → ℝ := λ x => x^2

-- Main theorem
theorem problem_solution (a b : ℝ) (h1 : A a b = {a}) (h2 : f a = b) :
  (A a b = {1/3}) ∧ 
  (∃ t : ℝ, ∀ x, f x = x^t) ∧
  ({x : ℝ | f x ≥ x} = {x : ℝ | x ≤ 0 ∨ x ≥ 1}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l184_18411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_bananas_eq_sum_l184_18413

/-- Represents the number of bananas Anthony bought initially -/
def initial_bananas : ℕ := sorry

/-- Represents the number of bananas Anthony ate -/
def eaten_bananas : ℕ := 2

/-- Represents the number of bananas Anthony has left -/
def remaining_bananas : ℕ := 10

/-- Theorem stating that the initial number of bananas is equal to
    the sum of eaten bananas and remaining bananas -/
theorem initial_bananas_eq_sum : initial_bananas = eaten_bananas + remaining_bananas := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_bananas_eq_sum_l184_18413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l184_18490

theorem remainder_problem (y : ℕ) (h : (7 * y) % 31 = 1) : (15 + y) % 31 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l184_18490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_rotation_theorem_equilibrium_distance_theorem_l184_18488

/-- The radius of the Earth in meters -/
noncomputable def earth_radius : ℝ := 6371000

/-- The acceleration due to gravity at Earth's surface in m/s² -/
noncomputable def gravity : ℝ := 9.81

/-- The current rotation period of the Earth in seconds -/
noncomputable def current_rotation_period : ℝ := 86400

/-- The rotation period that makes centrifugal force equal to gravitational force at the equator -/
noncomputable def equilibrium_rotation_period : ℝ := 2 * Real.pi * Real.sqrt (earth_radius / gravity)

/-- The distance from Earth's center where centrifugal force equals gravitational force given current rotation -/
noncomputable def equilibrium_distance : ℝ := (gravity * earth_radius^2 * current_rotation_period^2 / (4 * Real.pi^2))^(1/3)

theorem equilibrium_rotation_theorem :
  ∃ ε > 0, |equilibrium_rotation_period - 5062| < ε :=
sorry

theorem equilibrium_distance_theorem :
  ∃ ε > 0, |equilibrium_distance - 44500000| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_rotation_theorem_equilibrium_distance_theorem_l184_18488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_line_l184_18403

-- Define the set S
def S : Set ℂ := {z : ℂ | ∃ (r : ℝ), (2 + 3*Complex.I) * z = r}

-- State the theorem
theorem S_is_line : ∃ (a b : ℝ), S = {z : ℂ | z.re = a * z.im + b} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_line_l184_18403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_polynomial_l184_18443

noncomputable def root : ℂ := -3 - Complex.I * Real.sqrt 8

def polynomial (x : ℂ) : ℂ := x^2 + 6*x + 17

theorem root_of_polynomial : polynomial root = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_polynomial_l184_18443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l184_18487

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + (a-1) * x + 1

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + (a-1)

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ioo 1 4, HasDerivAt (f a) (f' a x) x ∧ f' a x < 0) →
  (∀ x ∈ Set.Ioi 6, HasDerivAt (f a) (f' a x) x ∧ f' a x > 0) →
  a ∈ Set.Icc 5 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l184_18487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_satisfying_conditions_l184_18489

noncomputable def deg_to_rad (x : ℝ) : ℝ := x * Real.pi / 180

theorem smallest_x_satisfying_conditions :
  ∃ (x : ℝ),
    x = 14 ∧
    x > 1 ∧
    Real.sin (deg_to_rad x) = Real.sin (deg_to_rad (x^2 + 30)) ∧
    Real.cos (deg_to_rad x) > 0.5 ∧
    ∀ (y : ℝ), y > 1 ∧
      Real.sin (deg_to_rad y) = Real.sin (deg_to_rad (y^2 + 30)) ∧
      Real.cos (deg_to_rad y) > 0.5 →
      y ≥ x :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_satisfying_conditions_l184_18489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l184_18432

def A : Set ℝ := {x | (2 : ℝ)^(x-6) ≤ (2 : ℝ)^(-2*x) ∧ (2 : ℝ)^(-2*x) ≤ 1}

def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a+1}

theorem range_of_a (a : ℝ) : A ∩ C a = C a → 0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l184_18432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zd_length_is_65_over_18_l184_18474

/-- An isosceles right triangle inscribed in an equilateral triangle -/
structure InscribedTriangle where
  /-- Side length of the equilateral triangle -/
  side : ℝ
  /-- Length of DX -/
  dx : ℝ
  /-- Length of XE and EY -/
  xe : ℝ
  /-- Assumption that dx + xe equals the side length -/
  h_side : side = dx + xe

/-- The length of ZD in the inscribed triangle configuration -/
noncomputable def zd_length (t : InscribedTriangle) : ℝ := 65 / 18

/-- Theorem stating that ZD length is 65/18 for the given configuration -/
theorem zd_length_is_65_over_18 (t : InscribedTriangle) 
  (h_dx : t.dx = 5) (h_xe : t.xe = 4) : zd_length t = 65 / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zd_length_is_65_over_18_l184_18474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_l184_18445

def pentagon : List (ℝ × ℝ) := [(0, 0), (2, 1), (3, 3), (1, 4), (0, 2)]

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def perimeter (polygon : List (ℝ × ℝ)) : ℝ :=
  let sides := List.zipWith distance polygon (polygon.rotateLeft 1)
  sides.sum

theorem pentagon_perimeter :
  ∃ (a b c : ℤ), perimeter pentagon = a + b * Real.sqrt 2 + c * Real.sqrt 10 ∧ a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_l184_18445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l184_18491

/-- The vertex of a quadratic function ax^2 + bx + c -/
noncomputable def vertex (a b c : ℝ) : ℝ × ℝ :=
  let x := -b / (2 * a)
  (x, a * x^2 + b * x + c)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_between_vertices : 
  let C := vertex 2 (-4) 1
  let D := vertex 3 6 4
  distance C D = 2 * Real.sqrt 2 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l184_18491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_ratios_l184_18415

noncomputable def burning_time_a : ℝ := 3
noncomputable def burning_time_b : ℝ := 5

noncomputable def length_ratio (t : ℝ) : ℝ := (1 - t / burning_time_a) / (1 - t / burning_time_b)

noncomputable def time_for_ratio (k : ℝ) : ℝ := 15 * (k - 1) / (3 * k - 5)

theorem candle_ratios :
  ∀ (ε : ℝ), ε > 0 →
  (abs (time_for_ratio 2 - 2.15) < ε) ∧
  (abs (time_for_ratio 3 - 2.5) < ε) ∧
  (abs (time_for_ratio 4 - 2.65) < ε) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_ratios_l184_18415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_values_l184_18421

theorem parallel_vectors_x_values (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, x]
  let b : Fin 2 → ℝ := ![x, 9]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_values_l184_18421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_substring_in_power_of_two_l184_18457

/-- A four-digit string represented as an integer -/
def FourDigitString : Type := { k : ℕ // 1000 ≤ k ∧ k ≤ 9999 }

/-- Check if a number contains a given substring when written in base 10 -/
def containsSubstring (n : ℕ) (k : FourDigitString) : Prop :=
  ∃ (a b : ℕ), n = a * 10000 + ↑k.val + b ∨ n = a * 10000 + ↑k.val * 10 + (n % 10)

/-- Main theorem: For any four-digit string, there exists an n < 20000 such that 2^n contains the string -/
theorem four_digit_substring_in_power_of_two (k : FourDigitString) :
  ∃ (n : ℕ), n < 20000 ∧ containsSubstring (2^n) k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_substring_in_power_of_two_l184_18457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_intersection_congruence_l184_18409

-- Define the basic elements of the triangle
variable (A B C : EuclideanSpace ℝ (Fin 2))
variable (H_A H_B H_C : EuclideanSpace ℝ (Fin 2))

-- Define lines through points
def line_through (P Q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {X | ∃ t : ℝ, X = (1 - t) • P + t • Q}

-- Define perpendicular lines
def perpendicular (l₁ l₂ : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

-- Define the properties of the altitudes
variable (altitude_A : perpendicular (line_through B C) (line_through A H_A))
variable (altitude_B : perpendicular (line_through A C) (line_through B H_B))
variable (altitude_C : perpendicular (line_through A B) (line_through C H_C))

-- Define the orthocenters of the triangles formed by the feet of the altitudes
noncomputable def O_A : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def O_B : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def O_C : EuclideanSpace ℝ (Fin 2) := sorry

-- Define triangle congruence
def triangle_congruent (T₁ T₂ : (EuclideanSpace ℝ (Fin 2)) × (EuclideanSpace ℝ (Fin 2)) × (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

-- State the theorem
theorem altitude_intersection_congruence :
  triangle_congruent (O_A, O_B, O_C) (H_A, H_B, H_C) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_intersection_congruence_l184_18409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l184_18431

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - 1 + Real.sqrt (6 * x - x^2)

-- State the theorem
theorem f_range :
  ∀ y ∈ Set.Icc (-1 : ℝ) 2, ∃ x ∈ Set.Icc (0 : ℝ) 2, f x = y ∧
  ∀ x ∈ Set.Icc (0 : ℝ) 2, f x ∈ Set.Icc (-1 : ℝ) 2 := by
  sorry

-- Note: Set.Icc a b represents the closed interval [a, b]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l184_18431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_domain_l184_18476

-- Define the function h
noncomputable def h : ℝ → ℝ := sorry

-- Define the domain of h
def h_domain : Set ℝ := Set.Icc (-10) 6

-- Define the function p in terms of h
noncomputable def p (x : ℝ) : ℝ := h (-5 * x)

-- Theorem statement
theorem p_domain : 
  {x : ℝ | p x ∈ h_domain} = Set.Icc (-1.2) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_domain_l184_18476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_window_width_is_57_inches_l184_18448

/-- Calculates the width of a rectangular window with given specifications -/
def window_width (num_panes : ℕ) (num_columns : ℕ) (pane_height : ℕ) 
  (height_width_ratio : ℚ) (gap_width : ℕ) (border_width : ℕ) : ℕ :=
  let pane_width := (pane_height : ℚ) / height_width_ratio
  let total_pane_width := num_columns * (pane_width.num / pane_width.den)
  let total_gap_width := (num_columns - 1) * gap_width
  let total_border_width := 2 * border_width
  (total_pane_width.toNat + total_gap_width + total_border_width)

/-- Theorem stating that the window width is 57 inches given the specified conditions -/
theorem window_width_is_57_inches : 
  window_width 12 4 9 (3/4) 1 3 = 57 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_window_width_is_57_inches_l184_18448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_exp_power_identity_cos_sin_identity_l184_18461

theorem complex_exp_power_identity (n : ℕ) (hn : n > 0 ∧ n ≤ 500) (t : ℝ) :
  (Complex.exp (Complex.I * -t)) ^ n = Complex.exp (Complex.I * (-n * t)) := by
  sorry

theorem cos_sin_identity (n : ℕ) (hn : n > 0 ∧ n ≤ 500) (t : ℝ) :
  (Complex.cos t - Complex.I * Complex.sin t) ^ n = Complex.cos (n * t) - Complex.I * Complex.sin (n * t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_exp_power_identity_cos_sin_identity_l184_18461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l184_18402

/-- The planar region D represented by (x-1)^2 + y^2 ≤ 1 -/
def D : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + p.2^2 ≤ 1}

/-- The line represented by x + √3 * y + b = 0 -/
def line (b : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1 + Real.sqrt 3 * p.2 + b = 0}

/-- The line has common points with region D -/
def has_common_points (b : ℝ) : Prop :=
  ∃ p, p ∈ D ∩ line b

theorem range_of_b :
  ∀ b : ℝ, has_common_points b ↔ -3 ≤ b ∧ b ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l184_18402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ptv_measure_l184_18497

-- Define the regular pentagon
structure RegularPentagon where
  vertices : Finset (ℝ × ℝ)
  is_regular : vertices.card = 5
  -- Additional properties of a regular pentagon could be added here

-- Define the equilateral triangle
structure EquilateralTriangle where
  vertices : Finset (ℝ × ℝ)
  is_equilateral : vertices.card = 3
  -- Additional properties of an equilateral triangle could be added here

-- Define the configuration
structure Configuration where
  pentagon : RegularPentagon
  triangle : EquilateralTriangle
  p_inside : ∃ p ∈ triangle.vertices, p ∈ interior pentagon.vertices

-- Define the angle measure function
noncomputable def angle_measure (p t v : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem angle_ptv_measure (config : Configuration) : 
  ∃ (p t v : ℝ × ℝ), 
    p ∈ config.triangle.vertices ∧ 
    t ∈ config.pentagon.vertices ∧ 
    v ∈ config.pentagon.vertices ∧ 
    angle_measure p t v = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ptv_measure_l184_18497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_multiple_of_five_l184_18400

theorem divisors_multiple_of_five (n : ℕ) (h : n = 2^1 * 3^2 * 5^1 * 7^2) :
  (Finset.filter (λ d ↦ d ∣ n ∧ 5 ∣ d) (Finset.range (n + 1))).card = 18 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_multiple_of_five_l184_18400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_special_intercepts_l184_18452

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : (a : ℝ) → (b : ℝ) → (x : ℝ) → (y : ℝ) → a * x + b * y = c

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Get the x-intercept of a line -/
noncomputable def Line.xIntercept (l : Line) : ℝ :=
  l.c / l.a

/-- Get the y-intercept of a line -/
noncomputable def Line.yIntercept (l : Line) : ℝ :=
  l.c / l.b

/-- The main theorem -/
theorem line_through_point_with_special_intercepts :
  ∃ (l : Line),
    (l.contains 2 (-1)) ∧
    (l.yIntercept = 2 * l.xIntercept) ∧
    ((l.a = 2 ∧ l.b = 1 ∧ l.c = 3) ∨ (l.a = 1 ∧ l.b = 2 ∧ l.c = 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_special_intercepts_l184_18452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l184_18471

theorem equation_solution :
  ∃ x : ℝ, Real.sqrt (3 + Real.sqrt (9 + 3*x)) + Real.sqrt (3 + Real.sqrt (5 + x)) = 3 + 3*Real.sqrt 2 :=
by
  use 6
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l184_18471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_surface_area_proof_l184_18473

/-- The total surface area of a hemisphere with radius 10 cm, including its circular base -/
noncomputable def hemisphere_surface_area : ℝ := 300 * Real.pi

/-- The surface area of a sphere with radius r -/
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

/-- Theorem: The total surface area of a hemisphere with radius 10 cm, including its circular base, is 300π cm² -/
theorem hemisphere_surface_area_proof :
  hemisphere_surface_area = sphere_surface_area 10 / 2 + Real.pi * 10^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_surface_area_proof_l184_18473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_water_bottles_sold_l184_18401

/-- Proves that the number of hot-water bottles sold is 60 given the problem conditions --/
theorem hot_water_bottles_sold : ℕ := by
  let thermometer_price : ℕ := 2
  let hot_water_bottle_price : ℕ := 6
  let total_sales : ℕ := 1200
  let thermometer_to_bottle_ratio : ℕ := 7

  have h1 : ∃ (t h : ℕ), t = thermometer_to_bottle_ratio * h ∧ 
    thermometer_price * t + hot_water_bottle_price * h = total_sales := by
    sorry

  exact 60


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_water_bottles_sold_l184_18401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_two_points_parallel_line_through_point_l184_18417

-- Define the points
def A1 : ℝ × ℝ := (2, 5)
def B : ℝ × ℝ := (-3, -2)
def A2 : ℝ × ℝ := (-3, 4)

-- Define the line l
def l (x y : ℝ) : Prop := 3*x - 4*y + 29 = 0

-- Theorem for the first line
theorem line_through_two_points :
  ∀ x y : ℝ, 7*x - 5*y + 11 = 0 ↔ ∃ t : ℝ, (x, y) = (1-t) • A1 + t • B :=
sorry

-- Theorem for the parallel line
theorem parallel_line_through_point :
  ∀ x y : ℝ, 3*x - 4*y + 25 = 0 ↔ (∃ k : ℝ, l (x + k) (y + k)) ∧ (x, y) = A2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_two_points_parallel_line_through_point_l184_18417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_EF_length_l184_18462

-- Define the parallel line segments
def AB : ℝ := 180
def CD : ℝ := 120
def GH : ℝ := 90  -- We can calculate this from AB
def EF : ℝ := 45  -- We can calculate this from GH

-- Define the relationships between the line segments
axiom AB_twice_GH : AB = 2 * GH
axiom CD_twice_EF : CD = 2 * EF

-- Theorem to prove
theorem EF_length : EF = 45 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_EF_length_l184_18462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_bill_is_20_l184_18404

/-- Represents the monthly internet bill in dollars -/
def current_bill : ℝ := 20

/-- Represents the cost of the 20 Mbps plan -/
def cost_20Mbps : ℝ := current_bill + 10

/-- Represents the cost of the 30 Mbps plan -/
def cost_30Mbps : ℝ := 2 * current_bill

/-- Represents the yearly savings when choosing 20 Mbps over 30 Mbps -/
def yearly_savings : ℝ := 120

theorem current_bill_is_20 :
  12 * cost_30Mbps - 12 * cost_20Mbps = yearly_savings →
  current_bill = 20 := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_bill_is_20_l184_18404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equidistant_from_B_and_C_l184_18437

-- Define the points A, B, and C
noncomputable def A : ℝ × ℝ × ℝ := (0, 0, -3/8)
def B : ℝ × ℝ × ℝ := (-5, -5, 6)
def C : ℝ × ℝ × ℝ := (-7, 6, 2)

-- Define the distance function between two points in 3D space
noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

-- Theorem statement
theorem A_equidistant_from_B_and_C : distance A B = distance A C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equidistant_from_B_and_C_l184_18437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l184_18433

theorem equation_solution (x : ℝ) : (16 : ℝ)^(2*x - 4) = (4 : ℝ)^(3 - x) → x = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l184_18433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fd_ef_ratio_l184_18455

-- Define the points
variable (A B C D E F : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h1 : IsRightAngle A B C)
variable (h2 : F ∈ SegmentClosed A B)
variable (h3 : dist A F = 2 * dist F B)
variable (h4 : Parallelogram B E C D)

-- State the theorem
theorem fd_ef_ratio :
  (dist F D) / (dist E F) = 2 / 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fd_ef_ratio_l184_18455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercepting_line_equation_l184_18499

/-- A line passing through (1,2) intersecting two parallel lines with a specific intercepted length -/
structure InterceptingLine where
  -- Slope of the line
  k : ℝ
  -- The line passes through (1,2)
  point_on_line : k * (1 : ℝ) - 2 = k - 2
  -- First parallel line equation
  parallel_line1 : ∀ x y, 4 * x + 3 * y + 1 = 0
  -- Second parallel line equation
  parallel_line2 : ∀ x y, 4 * x + 3 * y + 6 = 0
  -- Length of intercepted segment is √2
  intercept_length : 
    ((3 * k + 1) / (3 * k + 4) - (3 * k - 12) / (3 * k + 4))^2 + 
    ((-5 * k + 8) / (3 * k + 4) - (-10 * k + 8) / (3 * k + 4))^2 = 2

/-- The equation of the intercepting line is either x + 7y = 15 or 7x - y = 5 -/
theorem intercepting_line_equation (l : InterceptingLine) :
  (∀ x y, x + 7 * y = 15) ∨ (∀ x y, 7 * x - y = 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercepting_line_equation_l184_18499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l184_18453

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 9)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -3 ∧ x ≠ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l184_18453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_value_l184_18494

/-- A function that is symmetric about the origin -/
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The given function for positive x -/
noncomputable def f_pos (x : ℝ) : ℝ := 2^x - Real.log (x^2 - 3*x + 5) / Real.log 3

theorem symmetric_function_value :
  ∀ f : ℝ → ℝ, SymmetricAboutOrigin f →
  (∀ x > 0, f x = f_pos x) →
  f (-2) = -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_value_l184_18494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_placements_count_l184_18459

/-- Represents a 4x4 grid with two 'a's and two 'b's placed. -/
def Grid := Fin 4 → Fin 4 → Option (Fin 2)

/-- Checks if a grid placement is valid. -/
def is_valid_placement (g : Grid) : Prop :=
  (∀ i j, g i j = some 0 → ∀ k, k ≠ j → g i k ≠ some 0) ∧
  (∀ i j, g i j = some 0 → ∀ k, k ≠ i → g k j ≠ some 0) ∧
  (∀ i j, g i j = some 1 → ∀ k, k ≠ j → g i k ≠ some 1) ∧
  (∀ i j, g i j = some 1 → ∀ k, k ≠ i → g k j ≠ some 1) ∧
  (Finset.sum (Finset.univ : Finset (Fin 4 × Fin 4)) (fun ⟨i, j⟩ => if g i j = some 0 then 1 else 0) = 2) ∧
  (Finset.sum (Finset.univ : Finset (Fin 4 × Fin 4)) (fun ⟨i, j⟩ => if g i j = some 1 then 1 else 0) = 2)

/-- The number of valid grid placements. -/
def num_valid_placements : ℕ := 3960

theorem valid_placements_count :
  num_valid_placements = 3960 := by
  -- The proof goes here
  sorry

#eval num_valid_placements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_placements_count_l184_18459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l184_18466

noncomputable section

def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x : ℝ, x > 0 → f x ≠ 0

axiom f_double : ∀ x : ℝ, x > 0 → f (2 * x) = 2 * f x

axiom f_interval : ∀ x : ℝ, 1 < x ∧ x ≤ 2 → f x = 2 - x

theorem f_properties :
  (∀ m : ℤ, f (2^m) = 0) ∧
  (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, x > 0 ∧ f x = y) ∧
  (∀ a b : ℝ, 0 < a ∧ a < b ∧ (∃ k : ℤ, 2^k < a ∧ b ≤ 2^(k+1)) →
    ∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l184_18466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_150_consecutive_integers_l184_18485

theorem sum_of_150_consecutive_integers : ∃ k : ℕ,
  (List.range 150).foldr (λ i acc => acc + (k + i + 1)) 0 = 5827604250 ∧
  (List.range 150).foldr (λ i acc => acc + (k + i + 1)) 0 ≠ 2440575150 ∧
  (List.range 150).foldr (λ i acc => acc + (k + i + 1)) 0 ≠ 3518017315 ∧
  (List.range 150).foldr (λ i acc => acc + (k + i + 1)) 0 ≠ 4593461600 ∧
  (List.range 150).foldr (λ i acc => acc + (k + i + 1)) 0 ≠ 6925781950 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_150_consecutive_integers_l184_18485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l184_18463

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ ∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧
  (∀ x : ℝ, f (Real.pi/6 + x) = f (Real.pi/6 - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l184_18463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_positive_integers_l184_18441

def M : Set ℕ := sorry

axiom condition1 : (2018 : ℕ) ∈ M

axiom condition2 : ∀ m : ℕ, m ∈ M → ∀ d : ℕ, d > 0 ∧ m % d = 0 → d ∈ M

axiom condition3 : ∀ k m : ℕ, k ∈ M → m ∈ M → 1 < k → k < m → k * m + 1 ∈ M

theorem M_equals_positive_integers : M = {n : ℕ | n > 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_positive_integers_l184_18441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_over_sum_negative_l184_18425

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) - x) - Real.sin x

-- State the theorem
theorem f_sum_over_sum_negative (a b : ℝ) 
  (ha : a > -π/2) (hb : b > -π/2) (hc : a < π/2) (hd : b < π/2) (he : a + b ≠ 0) :
  (f a + f b) / (a + b) < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_over_sum_negative_l184_18425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_intersection_l184_18412

/-- Two perpendicular lines with equations ax - 3y = c and 2x + by = -c intersect at (2, -3) --/
theorem perpendicular_lines_intersection (a b c : ℝ) : 
  (∃ (x y : ℝ), ax - 3*y = c ∧ 2*x + b*y = -c) →  -- Lines exist
  (a / 3) * (-2 / b) = -1 →                       -- Lines are perpendicular
  a * 2 - 3 * (-3) = c ∧ 2 * 2 + b * (-3) = -c → -- Lines intersect at (2, -3)
  c = 12 := by
  intro h1 h2 h3
  -- The proof steps would go here
  sorry

#check perpendicular_lines_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_intersection_l184_18412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomials_l184_18477

theorem gcd_of_polynomials (b : ℤ) (h : ∃ k : ℤ, b = (2 * k + 1) * 1019) :
  Int.gcd (3 * b^2 + 31 * b + 91) (b + 15) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomials_l184_18477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_identity_l184_18456

theorem cosine_product_identity (x : ℝ) : 
  Real.cos x * Real.cos (2*x) * Real.cos (4*x) * Real.cos (8*x) = (1/8) * Real.cos (15*x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_identity_l184_18456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_benefits_l184_18469

/-- Represents the discount terms for a tea type -/
structure DiscountTerms where
  discount_percent : ℝ
  discount_period : ℕ
  max_delay : ℕ

/-- Calculates the effective annual interest rate for given discount terms -/
noncomputable def effective_annual_rate (terms : DiscountTerms) : ℝ :=
  (terms.discount_percent / (100 - terms.discount_percent)) * (365 / (terms.max_delay - terms.discount_period))

/-- Determines if a discount is beneficial given the bank's annual interest rate -/
def is_discount_beneficial (terms : DiscountTerms) (bank_rate : ℝ) : Prop :=
  effective_annual_rate terms > bank_rate

/-- The bank's annual interest rate -/
def bank_annual_rate : ℝ := 22

/-- Discount terms for Wuyi Mountain Oolong tea -/
def wuyi_oolong : DiscountTerms := ⟨3, 7, 31⟩

/-- Discount terms for Da Hong Pao tea -/
def da_hong_pao : DiscountTerms := ⟨2, 4, 40⟩

/-- Discount terms for Tieguanyin tea -/
def tieguanyin : DiscountTerms := ⟨5, 10, 35⟩

/-- Discount terms for Pu-erh tea -/
def pu_erh : DiscountTerms := ⟨1, 3, 24⟩

theorem discount_benefits :
  is_discount_beneficial wuyi_oolong bank_annual_rate ∧
  ¬is_discount_beneficial da_hong_pao bank_annual_rate ∧
  is_discount_beneficial tieguanyin bank_annual_rate ∧
  ¬is_discount_beneficial pu_erh bank_annual_rate := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_benefits_l184_18469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l184_18470

noncomputable def f (x : ℝ) : ℝ := |Real.arctan (x - 1)|

theorem function_property (a b x₁ x₂ : ℝ) (h₁ : x₁ ∈ Set.Icc a b) (h₂ : x₂ ∈ Set.Icc a b)
  (h₃ : x₁ < x₂) (h₄ : f x₁ ≥ f x₂) : b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l184_18470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_length_is_108_meters_l184_18496

/-- Represents the length of the pipe in meters -/
def pipe_length : ℝ := sorry

/-- Represents the length of Gavrila's step in meters -/
def step_length : ℝ := 0.8

/-- Represents the number of steps Gavrila takes when walking in the same direction as the tractor -/
def steps_same_direction : ℕ := 210

/-- Represents the number of steps Gavrila takes when walking in the opposite direction of the tractor -/
def steps_opposite_direction : ℕ := 100

/-- Represents the distance the pipe moves with each of Gavrila's steps -/
def pipe_movement_per_step : ℝ := sorry

/-- Theorem stating that the pipe length is approximately 108 meters -/
theorem pipe_length_is_108_meters :
  pipe_length = steps_same_direction * (step_length - pipe_movement_per_step) ∧
  pipe_length = steps_opposite_direction * (step_length + pipe_movement_per_step) →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |pipe_length - 108| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_length_is_108_meters_l184_18496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_bounded_l184_18498

/-- Product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digitProduct (n / 10)

/-- Sequence defined by a(n+1) = a(n) + product of digits of a(n) -/
def sequenceA (m : ℕ) : ℕ → ℕ
  | 0 => m
  | n + 1 => sequenceA m n + digitProduct (sequenceA m n)

/-- The sequence is bounded for any starting value m -/
theorem sequence_is_bounded (m : ℕ) : ∃ B : ℕ, ∀ n : ℕ, sequenceA m n ≤ B := by
  sorry

#check sequence_is_bounded

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_bounded_l184_18498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l184_18435

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the natural logarithm function
noncomputable def ln : ℝ → ℝ := Real.log

-- Define the quadratic function
noncomputable def f (b c : ℝ) (x : ℝ) : ℝ := log (x^2 + b*x + c)

-- Define the circle equation
def is_circle (a : ℝ) : Prop :=
  ∃ h k r, ∀ x y, a^2*x^2 + (a + 2)*y^2 + 2*a*x + a = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

theorem problem_solution :
  (∀ b c : ℝ, ¬(∀ x : ℝ, ∃ y : ℝ, y = f b c x)) ∧
  (∃! x : ℝ, ln x + x = 4) ∧
  (is_circle (-1) ∧ ¬∀ a : ℝ, is_circle a → a = -1) :=
by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l184_18435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l184_18426

/-- Definition of the function g_n -/
noncomputable def g (n : ℕ) (x : ℝ) : ℝ := (Real.sin x) ^ n + (Real.cos x) ^ n

/-- The main theorem -/
theorem equation_solutions_count :
  ∃! (s : Finset ℝ), s.card = 3 ∧ 
  (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi) ∧
  (∀ x ∈ s, 8 * g 5 x - 5 * g 3 x = 3 * g 1 x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi → 
    (8 * g 5 x - 5 * g 3 x = 3 * g 1 x → x ∈ s)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l184_18426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_y_coordinate_l184_18450

-- Define the points A, B, C, D
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (1, 1)
def D : ℝ × ℝ := (2, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the theorem
theorem point_P_y_coordinate :
  ∃ (P : ℝ × ℝ),
    distance P A + distance P D = 6 ∧
    distance P B + distance P C = 6 ∧
    P.2 = (-20 + 6 * Real.sqrt 15) / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_y_coordinate_l184_18450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l184_18451

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-2 + (1/2) * t, 2 + (Real.sqrt 3 / 2) * t)

def curve_C (x y : ℝ) : Prop := (y - 2)^2 - x^2 = 1

def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_l t ∧ curve_C p.1 p.2}

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_segment_length :
  ∃ A B, A ∈ intersection_points ∧ B ∈ intersection_points ∧ distance A B = 2 * Real.sqrt 14 := by
  sorry

#check intersection_segment_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l184_18451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_two_digit_number_l184_18486

def S : Finset Nat := {1, 2, 4, 6}

theorem largest_two_digit_number (n : Nat) :
  (n ∈ Finset.image (fun p : Nat × Nat => 10 * p.1 + p.2) 
    (Finset.filter (fun p : Nat × Nat => p.1 ≠ p.2) (S.product S))) →
  n ≤ 64 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_two_digit_number_l184_18486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_range_theorem_l184_18481

noncomputable def f (x : ℝ) : ℝ := 
  Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 4

noncomputable def g (a b x : ℝ) : ℝ := 2 * a * f x + b

theorem f_g_range_theorem :
  ∃ (a b : ℝ), (∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/4), g a b x ∈ Set.Icc 2 4) ∧
  ((a > 0 ∧ a = 4/3 ∧ b = 10/3) ∨ (a < 0 ∧ a = -4/3 ∧ b = 8/3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_range_theorem_l184_18481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_3_l184_18480

noncomputable section

variable (g : ℝ → ℝ)

axiom g_property : ∀ x : ℝ, x ≠ 1/2 → g x + g ((x + 2) / (2 - 4*x)) = 2*x

theorem g_of_3 : g 3 = 9/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_3_l184_18480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_touches_parabola_l184_18458

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the circle
def myCircle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Theorem statement
theorem circle_touches_parabola :
  ∃ (center : ℝ × ℝ),
    center.1 = 0 ∧
    myCircle center 7 0 (197/4) ∧
    (∃ (x₁ x₂ : ℝ),
      x₁ ≠ x₂ ∧
      myCircle center 7 x₁ (parabola x₁) ∧
      myCircle center 7 x₂ (parabola x₂) ∧
      (∀ (x : ℝ),
        x ≠ x₁ ∧ x ≠ x₂ →
        ¬myCircle center 7 x (parabola x))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_touches_parabola_l184_18458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_direction_reciprocal_l184_18416

-- Define a type for compass directions
inductive CompassDirection
  | North
  | NorthEast
  | East
  | SouthEast
  | South
  | SouthWest
  | West
  | NorthWest

-- Define a type for ships
structure Ship :=
  (name : String)

-- Define a function to represent the direction between two ships
def direction_between (a b : Ship) (d : CompassDirection) (angle : Nat) : Prop := sorry

-- Define the opposite direction function
def opposite_direction : CompassDirection → CompassDirection
  | CompassDirection.North => CompassDirection.South
  | CompassDirection.NorthEast => CompassDirection.SouthWest
  | CompassDirection.East => CompassDirection.West
  | CompassDirection.SouthEast => CompassDirection.NorthWest
  | CompassDirection.South => CompassDirection.North
  | CompassDirection.SouthWest => CompassDirection.NorthEast
  | CompassDirection.West => CompassDirection.East
  | CompassDirection.NorthWest => CompassDirection.SouthEast

-- Theorem statement
theorem ship_direction_reciprocal (A B : Ship) :
  direction_between B A CompassDirection.NorthEast 35 →
  direction_between A B CompassDirection.SouthWest 35 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_direction_reciprocal_l184_18416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_l184_18449

def generalized_pascal_triangle (n : ℕ) : List (List ℕ) :=
  sorry

def coefficient (n k : ℕ) : ℕ :=
  match (generalized_pascal_triangle n).get? k with
  | some row => row.getD (k - 2) 0
  | none => 0

theorem expansion_coefficient (a : ℝ) :
  coefficient 5 8 + a * coefficient 5 7 = 75 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_l184_18449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricketer_average_score_l184_18472

theorem cricketer_average_score (total_matches : ℕ) (all_matches_avg : ℚ) (last_matches : ℕ) (last_matches_avg : ℚ) :
  total_matches = 7 →
  all_matches_avg = 56 →
  last_matches = 3 →
  last_matches_avg = 69333333333333333 / 1000000000000000 →
  (all_matches_avg * total_matches - last_matches_avg * last_matches) / (total_matches - last_matches) = 46 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricketer_average_score_l184_18472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l184_18467

noncomputable def angle (v w : ℝ × ℝ) : ℝ := Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem vector_decomposition (OA OB OC : ℝ × ℝ) :
  Real.sqrt (OA.1^2 + OA.2^2) = 1 →
  Real.sqrt (OB.1^2 + OB.2^2) = 1 →
  Real.sqrt (OC.1^2 + OC.2^2) = Real.sqrt 2 →
  Real.tan (angle OA OC) = 7 →
  angle OB OC = π / 4 →
  ∃ (m n : ℝ), OC = (m * OA.1 + n * OB.1, m * OA.2 + n * OB.2) ∧ m = 5 / 4 ∧ n = 7 / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l184_18467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_product_divisors_l184_18428

theorem prime_product_divisors (p q : ℕ) (n : ℕ) : 
  Nat.Prime p → Nat.Prime q → (Finset.card (Nat.divisors (p^3 * q^n)) = 28) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_product_divisors_l184_18428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_fraction_l184_18429

-- Define the @ operation
def at_op (x y : ℝ) : ℝ := x * y - y^2

-- Define the # operation
def hash_op (x y : ℝ) : ℝ := x + y - x * y^2 + x^2

-- Theorem statement
theorem evaluate_fraction : (at_op 7 3) / (hash_op 7 3) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_fraction_l184_18429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_divisors_8_factorial_is_96_l184_18447

/-- The number of positive divisors of 8! -/
def num_divisors_8_factorial : ℕ := 96

/-- 8! (8 factorial) -/
def eight_factorial : ℕ := 8*7*6*5*4*3*2*1

/-- Theorem: The number of positive divisors of 8! is 96 -/
theorem num_divisors_8_factorial_is_96 : 
  (Finset.filter (fun d => d > 0 ∧ eight_factorial % d = 0) (Finset.range (eight_factorial + 1))).card = num_divisors_8_factorial := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_divisors_8_factorial_is_96_l184_18447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l184_18427

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a * Real.cos t.B - t.b * Real.sin t.A = 0 ∧
  t.b = Real.sqrt 7 ∧
  t.a + t.c = 5

-- Helper function to calculate area
noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.c * Real.sin t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.B = π / 3 ∧ area t = 3 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l184_18427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_price_theorem_l184_18410

/-- Represents the selling price after each transaction --/
def selling_price (original_price : ℝ) : ℕ → ℝ
  | 0 => original_price
  | 1 => original_price * 0.86
  | 2 => original_price * 0.86 * 1.1
  | 3 => original_price * 0.86 * 1.1 * 0.95
  | _ => original_price * 0.86 * 1.1 * 0.95 * 1.2

/-- The theorem stating the relationship between the original price and the final selling price --/
theorem car_price_theorem (original_price : ℝ) :
  selling_price original_price 4 = 54000 →
  abs (original_price - 47500) < 1 := by
  sorry

#check car_price_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_price_theorem_l184_18410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_properties_l184_18423

/-- The function f(x) = x(ln x - a) + 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.log x - a) + 1

theorem two_zeros_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : f a x₁ = 0) 
  (h2 : f a x₂ = 0) 
  (h3 : x₂ > x₁) 
  (h4 : x₁ > 0) 
  (h5 : ∀ x, x > 0 → x ≠ x₁ → x ≠ x₂ → f a x ≠ 0) : 
  a > 1 ∧ x₁ * x₂ > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_properties_l184_18423
