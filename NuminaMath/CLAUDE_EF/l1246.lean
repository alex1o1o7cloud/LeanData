import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_sum_is_1300_l1246_124678

/-- Given an interest rate and time period, calculates the simple interest factor -/
def simpleInterestFactor (rate : ℝ) (time : ℝ) : ℝ :=
  1 + (rate * time)

/-- Given an interest rate and time period, calculates the compound interest factor -/
noncomputable def compoundInterestFactor (rate : ℝ) (time : ℝ) : ℝ :=
  (1 + rate) ^ time

/-- Theorem stating that given the conditions, the principal sum is 1300 -/
theorem principal_sum_is_1300 (rate : ℝ) (time : ℝ) (difference : ℝ) :
  rate = 0.1 →
  time = 2 →
  difference = 13 →
  (λ p ↦ p * (compoundInterestFactor rate time - simpleInterestFactor rate time)) 1300 = difference :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_sum_is_1300_l1246_124678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_l1246_124633

theorem sin_shift (x : ℝ) : 
  Real.sin (2 * x + π / 12) = Real.sin (2 * (x + π / 24)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_l1246_124633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABC_l1246_124652

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define points D, E, and F
noncomputable def D (A B C : ℝ × ℝ) : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
noncomputable def E (A B C : ℝ × ℝ) : ℝ × ℝ := ((2 * A.1 + C.1) / 3, (2 * A.2 + C.2) / 3)
noncomputable def F (A B C : ℝ × ℝ) : ℝ × ℝ := 
  let D := D A B C
  ((3 * A.1 + D.1) / 4, (3 * A.2 + D.2) / 4)

-- Define the area function
noncomputable def area (p q r : ℝ × ℝ) : ℝ := 
  (1/2) * abs ((q.1 - p.1) * (r.2 - p.2) - (r.1 - p.1) * (q.2 - p.2))

-- State the theorem
theorem area_ABC (A B C : ℝ × ℝ) (h : area (D A B C) (E A B C) (F A B C) = 23) : 
  area A B C = 92 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABC_l1246_124652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evin_losing_positions_l1246_124681

theorem evin_losing_positions : 
  (Finset.filter (λ x : ℕ => x % 5 = 0 ∨ x % 5 = 1) (Finset.range 1435)).card = 573 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evin_losing_positions_l1246_124681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_w_squared_value_l1246_124691

theorem w_squared_value (w : ℝ) (h : (w + 15)^2 = (4*w + 9)*(3*w + 6)) :
  abs (w^2 - 3.101) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_w_squared_value_l1246_124691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_tangent_property_l1246_124636

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define the lines y = x and y = -x
def line_pos (x : ℝ) : ℝ × ℝ := (x, x)
def line_neg (x : ℝ) : ℝ × ℝ := (x, -x)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the vector from O to a point
def vector_OP (p : ℝ × ℝ) : ℝ × ℝ := p

-- Define the sum of two vectors
def vector_sum (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define the scalar multiplication of a vector
def vector_scale (s : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (s * v.1, s * v.2)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- State the theorem
theorem trajectory_and_tangent_property :
  ∀ (P : ℝ × ℝ) (a b : ℝ),
    (∃ (t : ℝ), line_pos t = (a, a) ∧ line_neg t = (b, -b)) →
    distance (a, a) (b, -b) = 4 * Real.sqrt 5 / 5 →
    vector_OP P = vector_scale (1/2) (vector_sum (vector_OP (a, a)) (vector_OP (b, -b))) →
    (P.1^2 + P.2^2 = 4/5) ∧
    (∀ (M N : ℝ × ℝ),
      (M.1^2 / 4 + M.2^2 = 1) →
      (N.1^2 / 4 + N.2^2 = 1) →
      (∃ (k m : ℝ), (M.2 = k * M.1 + m ∧ N.2 = k * N.1 + m) ∨ (M.1 = N.1)) →
      dot_product (vector_OP M) (vector_OP N) = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_tangent_property_l1246_124636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_l1246_124634

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.sin x) + Real.sqrt ((1 / 2) - Real.cos x)
noncomputable def g (x : ℝ) : ℝ := (Real.cos x)^2 - Real.sin x

-- Define the domain of f
def domain_f : Set ℝ := {x | ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 3 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi}

-- Define the interval for g
def interval_g : Set ℝ := {x | -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4}

-- Define the range of g
def range_g : Set ℝ := {y | (2 - 2 * Real.sqrt 2) / 4 ≤ y ∧ y ≤ 5 / 4}

-- Theorem for the domain of f
theorem domain_of_f : {x : ℝ | ∃ y, f x = y} = domain_f := by sorry

-- Theorem for the range of g
theorem range_of_g : {y : ℝ | ∃ x ∈ interval_g, g x = y} = range_g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_l1246_124634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increment_approximation_l1246_124670

noncomputable def f (x : ℝ) := x * Real.sqrt (x^2 + 5)

theorem function_increment_approximation :
  let x₀ : ℝ := 2
  let Δx : ℝ := 0.2
  let y₀ := f x₀
  let y₁ := f (x₀ + Δx)
  let Δy := y₁ - y₀
  abs (Δy - 0.87) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increment_approximation_l1246_124670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_glycerin_mixture_concentration_l1246_124689

/-- Represents a glycerin solution with a given volume and concentration -/
structure GlycerinSolution where
  volume : ℚ
  concentration : ℚ

/-- Calculates the total amount of glycerin in a solution -/
def glycerinAmount (solution : GlycerinSolution) : ℚ :=
  solution.volume * solution.concentration

/-- Calculates the percentage of glycerin in a mixture of two solutions -/
def mixedConcentration (sol1 sol2 : GlycerinSolution) : ℚ :=
  (glycerinAmount sol1 + glycerinAmount sol2) / (sol1.volume + sol2.volume)

theorem glycerin_mixture_concentration :
  let sol1 : GlycerinSolution := { volume := 75, concentration := 3/10 }
  let sol2 : GlycerinSolution := { volume := 75, concentration := 9/10 }
  mixedConcentration sol1 sol2 = 6/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_glycerin_mixture_concentration_l1246_124689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1000th_term_l1246_124612

def my_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 4014 ∧ a 2 = 4015 ∧ ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 2 * n

theorem sequence_1000th_term (a : ℕ → ℤ) (h : my_sequence a) : a 1000 = 4680 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1000th_term_l1246_124612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_plane_passengers_l1246_124645

/-- Proves that the number of passengers on the second plane is 60 --/
theorem second_plane_passengers : ∃ passengers_second : ℕ, 
  -- Define the number of planes
  let num_planes : ℕ := 3
  -- Define the speed of an empty plane
  let empty_speed : ℕ := 600
  -- Define the speed reduction per passenger
  let speed_reduction_per_passenger : ℕ := 2
  -- Define the number of passengers on the first plane
  let passengers_first : ℕ := 50
  -- Define the number of passengers on the third plane
  let passengers_third : ℕ := 40
  -- Define the average speed of all planes
  let average_speed : ℕ := 500
  -- Define the function to calculate the speed of a plane given its passengers
  let speed (passengers : ℕ) : ℕ := empty_speed - speed_reduction_per_passenger * passengers
  -- Define the equation for the average speed
  let average_speed_equation (p : ℕ) : Prop :=
    (speed passengers_first + speed p + speed passengers_third) / num_planes = average_speed
  -- The theorem statement
  average_speed_equation passengers_second ∧ passengers_second = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_plane_passengers_l1246_124645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_has_winning_strategy_l1246_124623

-- Define the chessboard
def Chessboard := Fin 8 × Fin 8

-- Define the distance between two squares
def kingDistance (a b : Chessboard) : Nat :=
  max (Int.natAbs (a.1.val - b.1.val)) (Int.natAbs (a.2.val - b.2.val))

-- Define the game state
structure GameState where
  whitePos : Chessboard
  blackPos : Chessboard
  currentPlayer : Bool -- true for White, false for Black

-- Define the initial game state
def initialState : GameState :=
  { whitePos := (0, 0), blackPos := (7, 7), currentPlayer := true }

-- Define a valid move
def isValidMove (s : GameState) (newPos : Chessboard) : Prop :=
  let currentPos := if s.currentPlayer then s.whitePos else s.blackPos
  let otherPos := if s.currentPlayer then s.blackPos else s.whitePos
  kingDistance currentPos newPos = 1 ∧
  kingDistance newPos otherPos ≤ kingDistance currentPos otherPos

-- Define the winning condition
def isWinningPosition (s : GameState) : Prop :=
  (s.currentPlayer ∧ (s.whitePos.1 = 7 ∨ s.whitePos.2 = 7)) ∨
  (¬s.currentPlayer ∧ (s.blackPos.1 = 0 ∨ s.blackPos.2 = 0))

-- Theorem: White has a winning strategy
theorem white_has_winning_strategy :
  ∃ (strategy : GameState → Chessboard),
    ∀ (game : ℕ → GameState),
      game 0 = initialState →
      (∀ n, (game (n + 1)).whitePos = if (game n).currentPlayer
                                    then strategy (game n)
                                    else (game n).whitePos) →
      (∀ n, (game (n + 1)).blackPos = if (game n).currentPlayer
                                    then (game n).blackPos
                                    else strategy (game n)) →
      (∀ n, isValidMove (game n) (strategy (game n))) →
      ∃ n, isWinningPosition (game n) ∧ (game n).currentPlayer = true :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_has_winning_strategy_l1246_124623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1246_124662

-- Define the conditions
def p (x a : ℝ) : Prop := |x - a| < 4
def q (x : ℝ) : Prop := -x^2 + 5*x - 6 > 0

-- Define the sufficient condition relationship
def sufficient_condition (p q : ℝ → Prop) : Prop :=
  ∀ x, q x → p x

-- Define the range of a
def range_of_a : Set ℝ := {a | a ∈ Set.Icc (-1) 6 ∧ (a ≠ -1 ∨ a ≠ 6)}

-- State the theorem
theorem a_range :
  (∃ x, q x) ∧  -- q is not empty
  (sufficient_condition (p · a) q) ∧ -- q is sufficient for p
  (¬ ∀ x, p x a → q x) →  -- q is not necessary for p
  ∀ a, a ∈ range_of_a ↔ ∃ x, p x a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1246_124662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_shadow_theorem_l1246_124617

/-- The length of the cube's edge in centimeters -/
noncomputable def cube_edge : ℝ := 2

/-- The area of the shadow cast by the cube, excluding the area under the cube, in square centimeters -/
noncomputable def shadow_area : ℝ := 147

/-- The height of the light source above the center of the cube's top face in centimeters -/
noncomputable def y : ℝ := (Real.sqrt 151 - 2) / 2

theorem cube_shadow_theorem :
  let total_shadow_area := shadow_area + cube_edge ^ 2
  let shadow_side := Real.sqrt total_shadow_area
  let half_cube_diagonal := Real.sqrt 2
  y = (shadow_side - cube_edge) / 2 →
  ⌊1000 * y⌋ = 5150 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_shadow_theorem_l1246_124617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_residues_l1246_124667

/-- A sequence of integers (a_n) where a_{n+1} = a_n^3 + a_n^2 for all n ≥ 0 -/
def sequence_a (a₀ : ℤ) : ℕ → ℤ
  | 0 => a₀
  | n + 1 => (sequence_a a₀ n)^3 + (sequence_a a₀ n)^2

/-- The set of distinct residues modulo 11 for a given sequence -/
def distinct_residues (a₀ : ℤ) : Finset ℤ :=
  Finset.image (λ n ↦ (sequence_a a₀ n) % 11) (Finset.range 12)

theorem max_distinct_residues :
  (∀ a₀ : ℤ, Finset.card (distinct_residues a₀) ≤ 3) ∧
  (∃ a₀ : ℤ, Finset.card (distinct_residues a₀) = 3) := by
  sorry

#eval Finset.card (distinct_residues 6)  -- Should output 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_residues_l1246_124667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sneakers_price_calculation_l1246_124686

def calculate_final_price (original_price : ℚ) (coupon : ℚ) (promo_discount : ℚ) (event_discount : ℚ) (membership_discount : ℚ) (sales_tax : ℚ) : ℚ :=
  let price_after_coupon := original_price - coupon
  let price_after_promo := price_after_coupon * (1 - promo_discount)
  let price_after_event := price_after_promo * (1 - event_discount)
  let price_after_membership := price_after_event * (1 - membership_discount)
  let final_price := price_after_membership * (1 + sales_tax)
  (final_price * 100).floor / 100

theorem sneakers_price_calculation :
  calculate_final_price 120 10 (5/100) (3/100) (10/100) (7/100) = 9761/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sneakers_price_calculation_l1246_124686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1246_124621

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def satisfies_equation (t : Triangle) : Prop :=
  (2 * t.b - t.c) / t.a = Real.cos t.C / Real.cos t.A

def forms_right_triangle (t : Triangle) (p : Real) : Prop :=
  (p * Real.sin t.A) ^ 2 = (Real.sin t.B) ^ 2 + (Real.sin t.C) ^ 2

-- Theorem statement
theorem triangle_properties (t : Triangle) (p : Real) 
  (h1 : satisfies_equation t) 
  (h2 : forms_right_triangle t p) : 
  t.A = Real.pi / 3 ∧ 1 < p ∧ p ≤ Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1246_124621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repaint_cost_l1246_124608

def room_length : ℝ := 4
def room_width : ℝ := 3
def room_height : ℝ := 3
def door_window_area : ℝ := 4.7
def paint_per_sqm : ℝ := 0.6
def paint_per_bucket : ℝ := 4.5
def cost_per_bucket : ℝ := 286

def wall_area : ℝ := 2 * (room_length * room_height + room_width * room_height + room_length * room_width) - door_window_area

def paint_needed : ℝ := wall_area * paint_per_sqm

noncomputable def buckets_needed : ℕ := Nat.ceil (paint_needed / paint_per_bucket)

theorem repaint_cost (cost : ℝ) : cost = cost_per_bucket * (buckets_needed : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repaint_cost_l1246_124608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_cube_volume_theorem_l1246_124649

/-- The volume of the set of points that are inside or within one unit of a cube with side length 6 -/
noncomputable def extended_cube_volume : ℝ := 432 + 19 * Real.pi

/-- The side length of the cube -/
def cube_side_length : ℝ := 6

theorem extended_cube_volume_theorem :
  extended_cube_volume = 
    cube_side_length ^ 3 + 
    6 * cube_side_length ^ 2 + 
    Real.pi + 
    3 * cube_side_length * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_cube_volume_theorem_l1246_124649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_135_deg_inclination_l1246_124642

/-- The slope of a line with an inclination angle of 135 degrees -/
def slope_135 : ℝ := -1

/-- A point on the line -/
def point : ℝ × ℝ := (3, 2)

/-- The equation of a line passing through a given point with a given slope -/
def line_equation (p : ℝ × ℝ) (m : ℝ) : ℝ → ℝ → Prop :=
  fun x y => y - p.2 = m * (x - p.1)

theorem line_through_point_with_135_deg_inclination :
  ∀ x y, line_equation point slope_135 x y ↔ x + y - 5 = 0 := by
  sorry

#check line_through_point_with_135_deg_inclination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_135_deg_inclination_l1246_124642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_l1246_124688

/-- Represents the number of fifth graders -/
def f : ℕ := 1  -- We assign a value to f to make it concrete

/-- The average number of minutes run per day by third graders -/
def third_grade_avg : ℚ := 14

/-- The average number of minutes run per day by fourth graders -/
def fourth_grade_avg : ℚ := 18

/-- The average number of minutes run per day by fifth graders -/
def fifth_grade_avg : ℚ := 11

/-- The number of third graders -/
def third_graders : ℕ := 12 * f

/-- The number of fourth graders -/
def fourth_graders : ℕ := 4 * f

/-- The number of fifth graders -/
def fifth_graders : ℕ := f

/-- The total number of students -/
def total_students : ℕ := third_graders + fourth_graders + fifth_graders

/-- The total minutes run by all students -/
def total_minutes : ℚ := third_grade_avg * (third_graders : ℚ) + fourth_grade_avg * (fourth_graders : ℚ) + fifth_grade_avg * (fifth_graders : ℚ)

/-- Theorem: The average number of minutes run per day by all students is 251/17 -/
theorem average_minutes_run : total_minutes / (total_students : ℚ) = 251 / 17 := by
  sorry

#eval total_minutes / (total_students : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_l1246_124688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1246_124687

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 3 = 3 ∧ ∀ n : ℕ, n ≥ 3 → a (n + 1) = a n + 2

theorem sequence_properties (a : ℕ → ℤ) (h : arithmetic_sequence a) :
  (a 2 + a 4 = 6) ∧ (∀ n : ℕ, n ≥ 2 → a n = 2 * n - 3) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1246_124687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1246_124694

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 9

-- Define the line
def my_line (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- Theorem statement
theorem chord_length :
  ∃ (a b c d : ℝ),
    my_circle a b ∧ my_circle c d ∧
    my_line a b ∧ my_line c d ∧
    a ≠ c ∧ b ≠ d ∧
    ((a - c)^2 + (b - d)^2) = 16 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1246_124694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_equals_one_l1246_124637

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2*x - 2/x - 2*Real.log x

-- Define the derivative of f
noncomputable def f_derivative (x : ℝ) : ℝ := 2 + 2/x^2 - 2/x

-- Theorem statement
theorem tangent_line_at_x_equals_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  (fun x y => y = m * (x - x₀) + y₀) = (fun x y => y = 2*x - 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_equals_one_l1246_124637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l1246_124601

theorem pyramid_volume (base_area height : ℝ) (h1 : base_area = 16) (h2 : height = 4) :
  (1 / 3 : ℝ) * base_area * height = 64 / 3 := by
  sorry

#check pyramid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l1246_124601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_49_values_l1246_124619

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 + 1

noncomputable def g (y : ℝ) : ℝ := 
  let x := Real.sqrt ((y - 1) / 4)  -- Inverse of f
  x^2 - x + 1

-- Theorem statement
theorem sum_of_g_49_values : 
  (g 49 + g 49) = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_49_values_l1246_124619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l1246_124661

open BigOperators
open Real
open Nat

/-- The sum of the infinite series ∑(n=1 to ∞) (n^3 + n^2 + n - 1) / (n + 3)! is equal to 1/3 -/
theorem infinite_series_sum : 
  ∑' (n : ℕ), (n^3 + n^2 + n - 1 : ℚ) / (↑(n + 3).factorial) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l1246_124661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_g_l1246_124654

noncomputable def f (x : ℝ) := 3 * Real.sin (4 * x + Real.pi / 6)

noncomputable def g (x : ℝ) := 3 * Real.sin (2 * x - Real.pi / 6)

theorem axis_of_symmetry_g :
  ∀ x : ℝ, g (Real.pi / 3 + x) = g (Real.pi / 3 - x) := by
  sorry

#check axis_of_symmetry_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_g_l1246_124654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_l1246_124614

/-- The probability of getting heads on a single toss of the unfair coin. -/
noncomputable def p_heads : ℝ := 3/4

/-- The number of times the coin is tossed. -/
def num_tosses : ℕ := 60

/-- The probability of getting an even number of heads after n tosses. -/
noncomputable def P (n : ℕ) : ℝ := 1/2 * (1 + (-1/2)^n)

/-- The main theorem: The probability of getting an even number of heads
    after 60 tosses of an unfair coin with 3/4 probability of heads. -/
theorem even_heads_probability :
  P num_tosses = 1/2 * (1 + 1/(2^60)) := by
  sorry

/-- Auxiliary lemma: The recurrence relation for P(n) -/
lemma P_recurrence (n : ℕ) :
  P (n + 1) = 3/4 - 1/2 * P n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_l1246_124614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_l1246_124677

noncomputable def f (x : ℝ) : ℝ := Real.log (4 - 3*x - x^2) / Real.log 0.5

theorem interval_of_increase (x : ℝ) :
  (4 - 3*x - x^2 > 0) →
  StrictMono f ↔ (x > -3/2 ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_l1246_124677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1246_124631

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.cos x

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem problem_solution (a : ℕ → ℝ) :
  arithmetic_sequence a (π / 8) →
  (f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) = 5 * π) →
  (f (a 3))^2 - a 1 * a 5 = 13 * π^2 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1246_124631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birds_in_tree_specific_bird_scenario_l1246_124603

/-- Given an initial number of birds in a tree and a number of birds that fly up to the tree,
    the total number of birds in the tree is equal to the sum of the initial number and the
    number of birds that flew up. -/
theorem birds_in_tree (initial_birds new_birds : ℕ) :
  initial_birds + new_birds = initial_birds + new_birds :=
by
  rfl

/-- In this specific scenario, there were initially 14 birds, and 21 new birds flew up. -/
theorem specific_bird_scenario :
  14 + 21 = 35 :=
by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birds_in_tree_specific_bird_scenario_l1246_124603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_surface_area_l1246_124632

/-- The total surface area of a right pyramid with a hexagonal base -/
theorem hexagonal_pyramid_surface_area :
  ∀ (side_length peak_height : ℝ),
  side_length = 8 →
  peak_height = 15 →
  ∃ (total_area : ℝ),
  total_area = 96 * Real.sqrt 3 + 376.8 ∧
  abs (total_area - 543.2) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_surface_area_l1246_124632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kat_boxing_sessions_l1246_124620

/-- Represents Kat's weekly training schedule -/
structure TrainingSchedule where
  strength_hours : ℚ
  boxing_session_hours : ℚ
  total_hours : ℚ

/-- Calculates the number of boxing sessions per week -/
def boxing_sessions (schedule : TrainingSchedule) : ℚ :=
  (schedule.total_hours - schedule.strength_hours) / schedule.boxing_session_hours

/-- Theorem stating that Kat's boxing sessions per week is 4 -/
theorem kat_boxing_sessions :
  let schedule : TrainingSchedule := {
    strength_hours := 3,
    boxing_session_hours := 3/2,
    total_hours := 9
  }
  boxing_sessions schedule = 4 := by
  -- Unfold the definition and perform the calculation
  unfold boxing_sessions
  -- Simplify the arithmetic
  simp [TrainingSchedule.strength_hours, TrainingSchedule.boxing_session_hours, TrainingSchedule.total_hours]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kat_boxing_sessions_l1246_124620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_range_l1246_124638

-- Define the functions
noncomputable def f (x : ℝ) := Real.exp x
def g (a : ℝ) (x : ℝ) := a * x^2 - a * x

-- State the theorem
theorem symmetric_points_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    x₁ ≠ x₂ ∧
    f x₁ = g a x₁ ∧
    f x₂ = g a x₂ ∧
    f x₁ = x₂ ∧
    f x₂ = x₁) ↔
  (a ∈ Set.Ioo 0 1 ∪ Set.Ioi 1) :=
by
  sorry

#check symmetric_points_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_range_l1246_124638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_g_l1246_124611

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

-- Define the shifted function g
noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 3)

-- Theorem statement
theorem symmetry_axis_of_g :
  ∀ x : ℝ, g (3 * Real.pi / 4 + x) = g (3 * Real.pi / 4 - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_g_l1246_124611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_l1246_124643

noncomputable def coefficient_of_term (expr : ℝ → ℝ → ℝ) (term : ℝ → ℝ → ℝ) : ℝ := 
  sorry

theorem expansion_coefficient (a : ℝ) : 
  (∃ (x y : ℝ), coefficient_of_term (fun x y => (x - 1/x) * (a + y)^6) (fun x y => x⁻¹*y^4) = -15) → 
  (a = 1 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_l1246_124643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joanne_main_job_hours_l1246_124693

/-- Represents Joanne's work schedule and earnings --/
structure WorkSchedule where
  main_job_rate : ℚ
  part_time_rate : ℚ
  part_time_hours : ℚ
  total_weekly_earnings : ℚ
  days_per_week : ℕ

/-- Calculates the number of hours worked at the main job each day --/
def main_job_hours (w : WorkSchedule) : ℚ :=
  (w.total_weekly_earnings - w.part_time_rate * w.part_time_hours * w.days_per_week) / 
  (w.main_job_rate * w.days_per_week)

/-- Theorem stating that Joanne works 8 hours at her main job each day --/
theorem joanne_main_job_hours :
  let w : WorkSchedule := {
    main_job_rate := 16
    part_time_rate := 27/2
    part_time_hours := 2
    total_weekly_earnings := 775
    days_per_week := 5
  }
  main_job_hours w = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joanne_main_job_hours_l1246_124693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_l1246_124630

/-- Given a boat that travels upstream and downstream, calculate the speed of the current. -/
theorem current_speed 
  (upstream_distance : ℝ) 
  (upstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : upstream_distance = 1) 
  (h2 : upstream_time = 25 / 60) 
  (h3 : downstream_distance = 1) 
  (h4 : downstream_time = 12 / 60) : 
  ∃ (current_speed : ℝ), abs (current_speed - 1.3) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_l1246_124630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_investment_l1246_124624

/-- Proof of total investment calculation --/
theorem total_investment
  (interest_rate_1 : ℝ)
  (interest_rate_2 : ℝ)
  (combined_interest : ℝ)
  (investment_2 : ℝ)
  (total_investment : ℝ)
  (h1 : interest_rate_1 = 0.065)
  (h2 : interest_rate_2 = 0.08)
  (h3 : combined_interest = 678.87)
  (h4 : investment_2 = 6258.0)
  (h5 : interest_rate_1 * (total_investment - investment_2) + interest_rate_2 * investment_2 = combined_interest) :
  total_investment = 9000.0 :=
by
  sorry

#check total_investment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_investment_l1246_124624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_sin_function_l1246_124647

/-- The minimum positive period of the function y = 3 * sin(2x + π/4) is π -/
theorem min_period_sin_function : 
  ∃ T : ℝ, T > 0 ∧ 
    (∀ x : ℝ, 3 * Real.sin (2 * (x + T) + π / 4) = 3 * Real.sin (2 * x + π / 4)) ∧ 
    (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, 3 * Real.sin (2 * (x + S) + π / 4) = 3 * Real.sin (2 * x + π / 4)) → T ≤ S) ∧ 
    T = π :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_sin_function_l1246_124647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_externally_tangent_l1246_124640

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : Circle) : Prop :=
  distance c1.center c2.center = c1.radius + c2.radius

/-- The first circle: x^2 + y^2 = 1 -/
def circle1 : Circle :=
  { center := (0, 0), radius := 1 }

/-- The second circle: x^2 + y^2 - 6x - 8y + 9 = 0 -/
def circle2 : Circle :=
  { center := (3, 4), radius := 4 }

/-- Theorem: The given circles are externally tangent -/
theorem circles_are_externally_tangent : externally_tangent circle1 circle2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_externally_tangent_l1246_124640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1246_124671

-- Define the parabola
def on_parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the distance function
noncomputable def distance_sum (x y : ℝ) : ℝ :=
  Real.sqrt ((x - 2)^2 + (y - 1)^2) + Real.sqrt ((x - 1)^2 + y^2)

-- Theorem statement
theorem min_distance_sum :
  ∀ x y : ℝ, on_parabola x y → distance_sum x y ≥ 3 :=
by
  sorry

#check min_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1246_124671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_one_equals_546_div_25_l1246_124676

-- Define the functions t and s
def t (x : ℝ) : ℝ := 5 * x - 10

noncomputable def s (y : ℝ) : ℝ := 
  let x := (y + 10) / 5  -- Inverse of t
  x^2 + 5 * x + 6

-- Theorem to prove
theorem s_of_one_equals_546_div_25 : s 1 = 546 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_one_equals_546_div_25_l1246_124676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_sum_l1246_124641

theorem pentagon_area_sum (u v : ℤ) (hu : u > 0) (hv : v > 0) (huv : v < u) : 
  let A := (u, v)
  let B := (v, u)
  let C := (-v, u)
  let D := (-v, -u)
  let E := (v, -u)
  let pentagon_area := u^2 + 3*u*v
  pentagon_area = 451 → u + v = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_sum_l1246_124641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hazel_hike_distance_l1246_124682

/-- Represents the distance walked in a given hour -/
def distance_walked (hour : Nat) : ℝ := sorry

/-- Represents whether Hazel took a break in a given hour -/
def took_break (hour : Nat) : Bool := sorry

theorem hazel_hike_distance :
  (distance_walked 1 = 2) →
  (took_break 2 = true) →
  (distance_walked 2 = 2 * distance_walked 1) →
  (distance_walked 3 = 0.5 * distance_walked 2) →
  (distance_walked 1 + distance_walked 2 + distance_walked 3 = 8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hazel_hike_distance_l1246_124682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_identity_l1246_124666

theorem sine_sum_identity (x : ℝ) (h : Real.sin (x + π / 6) = 1 / 3) :
  Real.sin (x - 5 * π / 6) + (Real.sin (π / 3 - x))^2 = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_identity_l1246_124666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_sum_squares_l1246_124696

theorem quadratic_roots_sum_squares (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 4 = 0 → 
  x₂^2 - 3*x₂ - 4 = 0 → 
  x₁^2 - 2*x₁*x₂ + x₂^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_sum_squares_l1246_124696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_plus_one_l1246_124626

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_plus_one (x : ℝ) : floor (x + 1) = floor x + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_plus_one_l1246_124626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1246_124644

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  2 * Real.sqrt a + 3 * (b ^ (1/3)) ≥ 5 * ((a * b) ^ (1/5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1246_124644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_estimate_l1246_124673

def BallDraw : Type := Fin 4

def isSuccessfulDraw (draw : List (Fin 4)) : Bool :=
  0 ∈ draw ∧ 1 ∈ draw ∧ draw.length = 3

def SimulationResult := List (List (Fin 4))

theorem probability_estimate (simulation : SimulationResult) 
  (h : simulation.length = 18) :
  (simulation.filter isSuccessfulDraw).length / simulation.length = 5 / 18 := by
  sorry

#check probability_estimate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_estimate_l1246_124673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_l1246_124610

-- Define a triangle with angles A, B, C
structure Triangle where
  A : Real
  B : Real
  C : Real

-- Define the properties of the triangle
def ValidTriangle (t : Triangle) : Prop :=
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi ∧
  t.A < t.B ∧ t.B < t.C ∧
  2 * t.B = t.A + t.C

-- Theorem statement
theorem angle_ratio (t : Triangle) (h : ValidTriangle t) :
  ∃ (k : Real), k > 0 ∧ t.A = k ∧ t.B = 2*k ∧ t.C = 3*k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_l1246_124610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1246_124639

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := (1/2) * Real.cos (2*x - φ)

noncomputable def g (x : ℝ) : ℝ := (1/2) * Real.cos (4*x - Real.pi/3)

theorem function_properties (φ : ℝ) :
  (0 < φ ∧ φ < Real.pi) →
  f φ (Real.pi/6) = 1/2 →
  φ = Real.pi/3 ∧
  Set.Icc (-1/4 : ℝ) (1/2) = Set.range (fun x => g x) ∩ Set.Icc 0 (Real.pi/4) :=
by sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1246_124639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1246_124675

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f x + f y) = y + (f x)^2

/-- The main theorem stating that any function satisfying the functional equation
    is either the identity function or the negation function -/
theorem functional_equation_solution (f : ℝ → ℝ) 
    (h : SatisfiesFunctionalEquation f) : 
    (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1246_124675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_pays_24_dollars_l1246_124683

/-- The cost of sausages for Jake --/
def jake_sausage_cost (package_weight : ℕ) (num_packages : ℕ) (price_per_pound : ℕ) : ℕ :=
  package_weight * num_packages * price_per_pound

/-- Proof that Jake pays $24 for the sausages --/
theorem jake_pays_24_dollars :
  jake_sausage_cost 2 3 4 = 24 := by
  -- Unfold the definition of jake_sausage_cost
  unfold jake_sausage_cost
  -- Perform the arithmetic
  norm_num

#eval jake_sausage_cost 2 3 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_pays_24_dollars_l1246_124683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_equal_l1246_124663

noncomputable def sequence_a (a₁ a₂ a₃ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | 1 => a₂
  | 2 => a₃
  | n + 3 => -1 / (sequence_a a₁ a₂ a₃ n + 1)

theorem sequence_sum_equal (a₁ a₂ a₃ : ℝ) 
  (h : a₁ + a₂ + a₃ = 6) :
  sequence_a a₁ a₂ a₃ 15 + sequence_a a₁ a₂ a₃ 16 + sequence_a a₁ a₂ a₃ 17 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_equal_l1246_124663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seminar_recording_distribution_l1246_124657

theorem seminar_recording_distribution (total_time : ℚ) (usb_capacity : ℚ) :
  total_time = 495 →
  usb_capacity = 65 →
  let num_usb := Int.ceil (total_time / usb_capacity)
  let audio_per_usb := total_time / num_usb
  audio_per_usb = 61.875 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seminar_recording_distribution_l1246_124657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_and_directrix_l1246_124635

/-- Given a parabola y = -1/6 x^2, prove its focus and directrix -/
theorem parabola_focus_and_directrix :
  let f : ℝ → ℝ := fun x ↦ -1/6 * x^2
  ∃ (focus : ℝ × ℝ) (directrix : ℝ → Prop),
    focus = (0, -3/2) ∧
    directrix = (fun y ↦ y = 3/2) ∧
    ∀ (x y : ℝ), f x = y ↔ 
      (x - focus.1)^2 + (y - focus.2)^2 = (y - 3/2)^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_and_directrix_l1246_124635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_with_common_chord_l1246_124656

noncomputable def circle_C1_center : ℝ × ℝ := (-2, 0)
noncomputable def circle_C2_center : ℝ × ℝ := (1, 3)
noncomputable def circle_radius : ℝ := Real.sqrt 10

theorem circles_intersect_with_common_chord :
  let distance_between_centers := Real.sqrt ((circle_C1_center.1 - circle_C2_center.1)^2 + 
                                             (circle_C1_center.2 - circle_C2_center.2)^2)
  let sum_of_radii := 2 * circle_radius
  let common_chord_length := Real.sqrt 22
  (distance_between_centers < sum_of_radii) ∧ 
  (common_chord_length^2 / 4 + (distance_between_centers / 2)^2 = circle_radius^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_with_common_chord_l1246_124656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_divides_angle_bisector_l1246_124690

-- Define a triangle with side lengths a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the incenter of the triangle
noncomputable def incenter (t : Triangle) : EuclideanSpace ℝ (Fin 3) := sorry

-- Define an angle bisector
noncomputable def angle_bisector (t : Triangle) (vertex : Fin 3) : Set (EuclideanSpace ℝ (Fin 3)) := sorry

-- Define a vertex of the triangle
noncomputable def Triangle.vertex (t : Triangle) (i : Fin 3) : EuclideanSpace ℝ (Fin 3) := sorry

-- Theorem stating the ratio property of the incenter on angle bisectors
theorem incenter_divides_angle_bisector (t : Triangle) :
  ∀ (vertex : Fin 3),
  let O := incenter t
  let bisector := angle_bisector t vertex
  let ratio := match vertex with
    | 0 => (t.b + t.c) / t.a
    | 1 => (t.c + t.a) / t.b
    | 2 => (t.a + t.b) / t.c
  ∃ (P : EuclideanSpace ℝ (Fin 3)),
  P ∈ bisector ∧
  dist O P / dist P (t.vertex vertex) = ratio := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_divides_angle_bisector_l1246_124690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_probability_l1246_124629

-- Define the urn state
structure UrnState where
  red : ℕ
  blue : ℕ

-- Define the operation
def operation (state : UrnState) (next_state : UrnState) : Prop :=
  (next_state.red = state.red + 1 ∧ next_state.blue = state.blue) ∨
  (next_state.red = state.red ∧ next_state.blue = state.blue + 1)

-- Define the initial state
def initial_state : UrnState :=
  ⟨1, 1⟩

-- Define the final state condition
def final_state (state : UrnState) : Prop :=
  state.red + state.blue = 7

-- Define the target state
def target_state : UrnState :=
  ⟨3, 4⟩

-- Define the number of operations
def num_operations : ℕ := 5

-- Define the probability of reaching the target state
def prob_target_state : ℚ := 1 / 6

-- Theorem statement
theorem urn_probability :
  ∀ (states : Fin (num_operations + 1) → UrnState),
    states 0 = initial_state
    → (∀ i : Fin num_operations, operation (states i) (states (i.succ)))
    → final_state (states num_operations)
    → (states num_operations = target_state) = (prob_target_state = 1 / 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_probability_l1246_124629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_y_value_l1246_124653

noncomputable def inverse_proportion (x : ℝ) : ℝ := 1 / x

theorem smallest_y_value (y₁ y₂ y₃ y₄ : ℝ) 
  (h₁ : y₁ = inverse_proportion (-2))
  (h₂ : y₂ = inverse_proportion (-1))
  (h₃ : y₃ = inverse_proportion 1)
  (h₄ : y₄ = inverse_proportion 2) :
  y₂ ≤ y₁ ∧ y₂ ≤ y₃ ∧ y₂ ≤ y₄ := by
  sorry

#check smallest_y_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_y_value_l1246_124653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_converse_false_l1246_124600

/-- A quadrilateral with four right angles -/
structure RightAngledQuadrilateral where
  angles_are_right : Bool

/-- A rectangle is a right-angled quadrilateral -/
structure Rectangle extends RightAngledQuadrilateral

/-- A square is a rectangle with equal sides -/
structure Square extends Rectangle where
  sides_equal : Bool

/-- Theorem: The converse of "All four angles of a square are right angles" is false -/
theorem square_converse_false :
  ∃ (q : RightAngledQuadrilateral), ¬(∃ (s : Square), s.toRightAngledQuadrilateral = q) := by
  sorry

#check square_converse_false

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_converse_false_l1246_124600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l1246_124684

/-- Given a function f, halving its abscissas, keeping ordinates unchanged, 
    and shifting π/3 units right results in sin(x - π/4) -/
theorem function_transformation (f : ℝ → ℝ) : 
  (∀ x, Real.sin (x - π/4) = f (x/2 - π/3)) → 
  (∀ x, f x = Real.sin (x/2 + π/12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l1246_124684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_l1246_124659

theorem points_on_line (u : ℝ) :
  (Real.sin u ^ 2) + (Real.cos u ^ 2) = 1 := by
  have h : Real.sin u ^ 2 + Real.cos u ^ 2 = 1 := Real.sin_sq_add_cos_sq u
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_l1246_124659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_proof_l1246_124650

/-- The fraction of the area shaded in the square division pattern -/
def shaded_fraction : ℚ := 12 / 13

/-- The ratio of the center four squares to the total area -/
def center_ratio : ℚ := 1 / 4

/-- The ratio of the outer ring to the total area in each subdivision -/
def outer_ring_ratio : ℚ := 3 / 4

/-- The geometric series representing the shaded area -/
def shaded_series (n : ℕ) : ℚ := 
  outer_ring_ratio * (1 - center_ratio^n) / (1 - center_ratio)

theorem shaded_area_proof : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |shaded_series n - shaded_fraction| < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_proof_l1246_124650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_number_set_l1246_124665

theorem existence_of_special_number_set : ∃ (S : Finset ℕ), 
  Finset.card S = 10 ∧ 
  (∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → ¬(a ∣ b)) ∧
  (∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → (a^2 ∣ b)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_number_set_l1246_124665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_rounded_to_tenth_l1246_124668

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The sum of 76.893 and 34.2176 rounded to the nearest tenth equals 111.1 -/
theorem sum_rounded_to_tenth : roundToNearestTenth (76.893 + 34.2176) = 111.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_rounded_to_tenth_l1246_124668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_radius_radius_circumscribed_circle_unit_hexagon_l1246_124680

/-- The radius of the circumscribed circle of a regular hexagon with given side length. -/
def radius_circumscribed_circle_regular_hexagon (side_length : ℝ) : ℝ := side_length

/-- The radius of the circumscribed circle of a regular hexagon is equal to its side length. -/
theorem regular_hexagon_radius (side_length : ℝ) : 
  side_length > 0 → radius_circumscribed_circle_regular_hexagon side_length = side_length := by
  intro h
  rfl

/-- The radius of the circumscribed circle of a regular hexagon with side length 1 is 1. -/
theorem radius_circumscribed_circle_unit_hexagon : 
  radius_circumscribed_circle_regular_hexagon 1 = 1 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_radius_radius_circumscribed_circle_unit_hexagon_l1246_124680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_own_inverse_l1246_124679

/-- The function g(x) = (3x + 2) / (mx - 5) -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (3 * x + 2) / (m * x - 5)

/-- Theorem stating that g is its own inverse if and only if m = 8 -/
theorem g_is_own_inverse (m : ℝ) : 
  (∀ x, g m (g m x) = x) ↔ m = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_own_inverse_l1246_124679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l1246_124660

theorem angle_in_third_quadrant (α : ℝ) 
  (h1 : Real.sin (2 * α) > 0) 
  (h2 : Real.cos α < 0) : 
  π < α ∧ α < 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l1246_124660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_six_l1246_124692

theorem sum_of_solutions_is_six : 
  ∃ (x₁ x₂ : ℝ), 
    ((2 : ℝ)^(x₁^2 - 4*x₁ + 1) = (4 : ℝ)^(x₁ - 3)) ∧ 
    ((2 : ℝ)^(x₂^2 - 4*x₂ + 1) = (4 : ℝ)^(x₂ - 3)) ∧ 
    x₁ ≠ x₂ ∧ 
    x₁ + x₂ = 6 ∧ 
    ∀ (x : ℝ), ((2 : ℝ)^(x^2 - 4*x + 1) = (4 : ℝ)^(x - 3)) → (x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_six_l1246_124692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_different_point_l1246_124699

-- Define the type for polar coordinates
def PolarCoord := ℝ × ℝ

-- Define the point M
noncomputable def M : PolarCoord := (-5, Real.pi/3)

-- Define the function to check if two polar coordinates represent the same point
def same_point (p q : PolarCoord) : Prop :=
  (p.1 = q.1 ∧ p.2 = q.2) ∨
  (p.1 = -q.1 ∧ (p.2 = q.2 + Real.pi ∨ p.2 = q.2 - Real.pi))

-- State the theorem
theorem unique_different_point :
  ¬(same_point M (5, -Real.pi/3)) ∧
  same_point M (5, 4*Real.pi/3) ∧
  same_point M (5, -2*Real.pi/3) ∧
  same_point M (-5, -5*Real.pi/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_different_point_l1246_124699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_eleven_range_l1246_124616

theorem divisible_by_eleven_range (n : ℕ) : 
  (∃ m : ℕ, m ≤ 79 ∧ 
   (∀ k : ℕ, k ∈ Finset.range 5 → (n + k * 11) % 11 = 0) ∧
   (∀ i : ℕ, i < n → ¬(∀ k : ℕ, k ∈ Finset.range 5 → (i + k * 11) % 11 = 0))) →
  n = 33 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_eleven_range_l1246_124616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_sikh_boys_l1246_124648

/-- Proves that the percentage of Sikh boys is 10% given the specified conditions --/
theorem percentage_of_sikh_boys 
  (total_boys : ℕ) 
  (muslim_percentage : ℚ) 
  (hindu_percentage : ℚ) 
  (other_boys : ℕ) 
  (h1 : total_boys = 850)
  (h2 : muslim_percentage = 46 / 100)
  (h3 : hindu_percentage = 28 / 100)
  (h4 : other_boys = 136) :
  (total_boys - (muslim_percentage * total_boys).floor - (hindu_percentage * total_boys).floor - other_boys : ℚ) / total_boys = 1 / 10 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_sikh_boys_l1246_124648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_arrangement_satisfies_condition_l1246_124655

/-- A regular 9-sided polygon -/
structure Nonagon where
  vertices : Fin 9 → ℕ

/-- Check if three vertices form an equilateral triangle in a nonagon -/
def is_equilateral_triangle (i j k : Fin 9) : Prop :=
  (j - i) % 9 = 3 ∧ (k - j) % 9 = 3 ∧ (i - k) % 9 = 3

/-- Check if one number is the arithmetic mean of the other two -/
def is_arithmetic_mean (a b c : ℕ) : Prop :=
  2 * b = a + c

/-- The arrangement of numbers on the nonagon -/
def arrangement : Nonagon :=
  ⟨λ i => 2016 + i⟩

/-- The main theorem -/
theorem consecutive_arrangement_satisfies_condition :
  ∀ i j k : Fin 9, is_equilateral_triangle i j k →
    is_arithmetic_mean (arrangement.vertices i) (arrangement.vertices j) (arrangement.vertices k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_arrangement_satisfies_condition_l1246_124655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_by_students_l1246_124674

/-- The price of a pencil in cents -/
def pencil_price : ℕ := 20

/-- The price of a pen in cents -/
def pen_price : ℕ := 50

/-- The price of a notebook in cents -/
def notebook_price : ℕ := 150

/-- Tolu's purchase quantities -/
def tolu_purchase : ℕ × ℕ × ℕ := (3, 2, 1)

/-- Robert's purchase quantities -/
def robert_purchase : ℕ × ℕ × ℕ := (5, 4, 2)

/-- Melissa's purchase quantities -/
def melissa_purchase : ℕ × ℕ × ℕ := (2, 3, 3)

/-- Calculate the total cost of a purchase in cents -/
def calculate_cost (purchase : ℕ × ℕ × ℕ) : ℕ :=
  purchase.1 * pencil_price + purchase.2.1 * pen_price + purchase.2.2 * notebook_price

/-- The total amount spent by all students in dollars -/
theorem total_spent_by_students : 
  (calculate_cost tolu_purchase + calculate_cost robert_purchase + calculate_cost melissa_purchase) / 100 = 1550 / 100 := by
  sorry

#eval (calculate_cost tolu_purchase + calculate_cost robert_purchase + calculate_cost melissa_purchase)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_by_students_l1246_124674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_l1246_124658

-- Define the points
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (1, 4)
def C : ℝ × ℝ := (-3, 2)

-- Define the reflection of C over the x-axis
def C'' : ℝ × ℝ := (-3, -2)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem reflection_distance : distance C C'' = 4 := by
  -- Unfold the definitions
  unfold distance C C''
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_l1246_124658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l1246_124609

/-- Defines an ellipse C with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 1/2

/-- Defines a point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the foci of the ellipse -/
noncomputable def foci (e : Ellipse) : Point × Point :=
  let c := (e.a^2 - e.b^2).sqrt
  (⟨-c, 0⟩, ⟨c, 0⟩)

/-- States the properties of point P -/
def point_properties (e : Ellipse) (p : Point) : Prop :=
  p.x^2 + p.y^2 = 25 ∧
  let (f1, f2) := foci e
  (p.x - f1.x) * (p.x - f2.x) + p.y^2 = 16

/-- Theorem stating the equation of the ellipse and existence of point M -/
theorem ellipse_and_fixed_point (e : Ellipse) (p : Point) 
  (h_p : point_properties e p) : 
  (∃ (x y : ℝ), x^2/18 + y^2/9 = 1) ∧ 
  (∃ (m : Point), m.x = 0 ∧ m.y = 3 ∧
    ∀ (k : ℝ), 
      let l := {(x, y) | y = k*x - 1}
      let intersections := {(x, y) | x^2/18 + y^2/9 = 1 ∧ y = k*x - 1}
      ∀ (a b : Point), 
        (a.x, a.y) ∈ intersections → 
        (b.x, b.y) ∈ intersections →
        a ≠ b →
        (m.x - (a.x + b.x)/2)^2 + (m.y - (a.y + b.y)/2)^2 = 
          ((a.x - b.x)^2 + (a.y - b.y)^2) / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l1246_124609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_power_value_l1246_124615

theorem log_power_value (a m n : ℝ) (ha : a > 0) (ha1 : a ≠ 1) 
  (hm : Real.logb a 2 = m) (hn : Real.logb a 3 = n) : 
  a^(2*m + n) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_power_value_l1246_124615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l1246_124627

theorem parallel_vectors_angle (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) -- α is an acute angle
  (h2 : (1/2 : ℝ) = Real.sin α * Real.sin α) -- vectors are parallel
  : α = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l1246_124627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l1246_124622

/-- Represents a trapezoid EFGH with specific properties -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  height : ℝ
  ef_eq_gh : EF = GH
  ef_length : EF = 10
  gh_length : GH = 20
  height_length : height = 5

/-- Calculates the perimeter of the trapezoid EFGH -/
noncomputable def perimeter (t : Trapezoid) : ℝ :=
  t.EF + t.GH + 2 * (Real.sqrt (t.height^2 + ((t.GH - t.EF) / 2)^2))

/-- Theorem stating that the perimeter of the given trapezoid is 30 + 10√2 -/
theorem trapezoid_perimeter (t : Trapezoid) : perimeter t = 30 + 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l1246_124622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_l1246_124664

/-- A triangular pyramid with mutually perpendicular lateral edges -/
structure TriangularPyramid where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The volume of a triangular pyramid with mutually perpendicular lateral edges -/
noncomputable def volume (p : TriangularPyramid) : ℝ := (1/6) * p.a * p.b * p.c

/-- Theorem: The volume of a triangular pyramid with mutually perpendicular lateral edges
    of lengths a, b, and c is (1/6)abc -/
theorem triangular_pyramid_volume (p : TriangularPyramid) :
  volume p = (1/6) * p.a * p.b * p.c := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_l1246_124664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l1246_124625

-- Define the angle α and point P
noncomputable def α : ℝ := Real.arccos (-4/5)
def P : ℝ × ℝ := (-8, 6)

-- Define the conditions
axiom origin_vertex : True  -- The vertex of angle α is at the origin
axiom initial_side : True   -- The initial side coincides with the non-negative half-axis of the x-axis
axiom terminal_side : P.1 = -8 ∧ P.2^2 = 36  -- The terminal side passes through P(-8, ±6)
axiom cos_alpha : Real.cos α = -4/5

-- Define the theorem
theorem angle_properties :
  (Real.tan α = 3/4 ∨ Real.tan α = -3/4) ∧
  ((2 * Real.cos (3 * Real.pi / 2 + α) + Real.cos (-α)) / 
   (Real.sin (5 * Real.pi / 2 - α) - Real.cos (Real.pi + α)) = 5/4 ∨
   (2 * Real.cos (3 * Real.pi / 2 + α) + Real.cos (-α)) / 
   (Real.sin (5 * Real.pi / 2 - α) - Real.cos (Real.pi + α)) = -1/4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l1246_124625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l1246_124628

def IsArithmeticSequence (s : List ℝ) : Prop :=
  s.length ≥ 3 ∧ 
  ∀ i : ℕ, i + 2 < s.length → 
    s[i+1]! = (s[i]! + s[i+2]!) / 2

theorem arithmetic_sequence_solution (x : ℝ) :
  x > 0 ∧ 
  IsArithmeticSequence [2^2, x^2, 4^2] →
  x = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l1246_124628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_primes_x_squared_plus_x_plus_one_l1246_124698

theorem infinite_primes_x_squared_plus_x_plus_one :
  ¬ (∃ n : ℕ, ∀ p : ℕ, Nat.Prime p → (∃ x y : ℕ+, (x : ℕ)^2 + x + 1 = p * y) → p ≤ n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_primes_x_squared_plus_x_plus_one_l1246_124698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_contraction_l1246_124697

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def is_composite_floor (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (f x) = floor x

theorem exists_non_contraction (f : ℝ → ℝ) (h : is_composite_floor f) :
  ∃ a b : ℝ, a ≠ b ∧ |f a - f b| ≥ |a - b| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_contraction_l1246_124697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1246_124604

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.cos (x + Real.pi/3)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1246_124604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_properties_l1246_124685

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

-- Define the inverse function of f
noncomputable def f_inverse : ℝ → ℝ := Function.invFun f

-- Theorem statement
theorem inverse_f_properties :
  (∀ x, f_inverse (-x) = -f_inverse x) ∧ 
  (∀ x y, 0 < x ∧ x < y → f_inverse x < f_inverse y) :=
by
  sorry

#check inverse_f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_properties_l1246_124685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_of_increasing_geometric_sequence_l1246_124606

/-- A geometric sequence with first term a and common ratio q -/
noncomputable def geometric_sequence (a : ℝ) (q : ℝ) : ℕ → ℝ := λ n ↦ a * q^(n - 1)

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem common_ratio_of_increasing_geometric_sequence 
  (a : ℝ) (q : ℝ) (h_inc : q > 1) 
  (h_a2 : geometric_sequence a q 2 = 2) 
  (h_s3 : geometric_sum a q 3 = 7) : 
  q = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_of_increasing_geometric_sequence_l1246_124606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_soccer_balls_buyable_l1246_124613

def volleyball_price : ℕ := 88
def soccer_ball_price : ℕ := 72
def volleyballs_bought : ℕ := 11

theorem max_soccer_balls_buyable : 
  (volleyball_price * volleyballs_bought) / soccer_ball_price = 13 := by
  norm_num
  sorry

#eval (volleyball_price * volleyballs_bought) / soccer_ball_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_soccer_balls_buyable_l1246_124613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1246_124669

open Real

noncomputable def f (x : ℝ) : ℝ := sin x - Real.sqrt 3 * cos x

theorem max_value_of_f :
  ∃ (x : ℝ), 0 ≤ x ∧ x < 2 * π ∧
  (∀ (y : ℝ), 0 ≤ y ∧ y < 2 * π → f y ≤ f x) ∧
  x = 5 * π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1246_124669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_side_length_exists_max_side_length_l1246_124695

/-- A triangle with three different integer side lengths and a perimeter of 24 units -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  different : a ≠ b ∧ b ≠ c ∧ a ≠ c
  perimeter : a + b + c = 24

/-- The maximum length of any side in the triangle is 11 -/
theorem max_side_length (t : Triangle) : t.a ≤ 11 ∧ t.b ≤ 11 ∧ t.c ≤ 11 := by
  sorry

/-- There exists a triangle satisfying the conditions with a side of length 11 -/
theorem exists_max_side_length : ∃ (t : Triangle), t.a = 11 ∨ t.b = 11 ∨ t.c = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_side_length_exists_max_side_length_l1246_124695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_equidistant_l1246_124646

/-- The distance between two points in 3D space -/
noncomputable def distance (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

/-- Point A coordinates -/
def A : ℝ × ℝ × ℝ := (-4, 1, 7)

/-- Point B coordinates -/
def B : ℝ × ℝ × ℝ := (3, 5, -2)

/-- Point C coordinates -/
noncomputable def C : ℝ × ℝ × ℝ := (0, 0, 14/9)

/-- Theorem: Point C is equidistant from points A and B -/
theorem point_C_equidistant : 
  distance A.1 A.2.1 A.2.2 C.1 C.2.1 C.2.2 = 
  distance B.1 B.2.1 B.2.2 C.1 C.2.1 C.2.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_equidistant_l1246_124646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_exp_neg_l1246_124607

-- Define the quadratic function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) > 0
def solution_set_f_pos : Set ℝ := {x | x < 1 ∨ x > Real.exp 1}

-- Theorem statement
theorem solution_set_f_exp_neg (h1 : ∀ x, f x > 0 ↔ x ∈ solution_set_f_pos) :
  {x : ℝ | f (Real.exp x) < 0} = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_exp_neg_l1246_124607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_work_time_l1246_124602

/-- The number of days A takes to finish the work -/
def A : ℝ := sorry

/-- The number of days B takes to finish the work -/
def B : ℝ := sorry

/-- B takes half the time of A to finish the work -/
axiom B_time : B = A / 2

/-- A and B together can finish 0.75 part of the work in a day -/
axiom work_rate : 1 / A + 1 / B = 0.75

theorem A_work_time : A = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_work_time_l1246_124602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_power_difference_as_sum_of_squares_l1246_124618

theorem fourth_power_difference_as_sum_of_squares :
  ∃ f : ℕ → ℕ, Function.Injective f ∧
    ∀ k : ℕ, ∃ x y : ℕ,
      (2 * (f k) ^ 2 + 1) ^ 4 - (2 * (f k) ^ 2) ^ 4 = x ^ 2 + y ^ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_power_difference_as_sum_of_squares_l1246_124618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_theorem_l1246_124651

/-- The ellipse C with left vertex at (-2, 0) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- The line l passing through (1, 0) -/
structure Line where
  m : ℝ

/-- Points P and Q where line l intersects ellipse C -/
structure IntersectionPoints where
  P : ℝ × ℝ
  Q : ℝ × ℝ

/-- The theorem statement -/
theorem ellipse_area_theorem (C : Ellipse) (l : Line) (PQ : IntersectionPoints) :
  (C.a = 2 ∧ C.b = Real.sqrt 3) ∧
  (∀ (x y : ℝ), (x^2 / C.a^2 + y^2 / C.b^2 = 1) ↔ (x^2 / 4 + y^2 / 3 = 1)) ∧
  (∃ (S : ℝ → ℝ), (∀ m, S m = 18 * Real.sqrt (1 + m^2) / (3 * m^2 + 4)) ∧
                   (∀ A, 0 < A ∧ A ≤ 9/2 → ∃ m, S m = A)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_theorem_l1246_124651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l1246_124605

theorem complex_equation_solution :
  ∀ z : ℂ, (1 - 2*Complex.I)*z = 1 + 2*Complex.I → z = -3/5 + 4/5*Complex.I := by
  intro z hypothesis
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l1246_124605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1246_124672

theorem remainder_problem (a b : ℕ) (h1 : a > b) (h2 : (a - b) % 6 = 5) : a % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1246_124672
