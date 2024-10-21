import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1287_128727

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (-1, 3)

noncomputable def proj_vector (v w : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (v.1 * w.1 + v.2 * w.2) / (w.1^2 + w.2^2)
  (scalar * w.1, scalar * w.2)

theorem projection_theorem :
  proj_vector a b = (-1/2, 3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1287_128727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1287_128736

-- Define the triangle ABC
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (4, 0)
def C (c : ℝ) : ℝ × ℝ := (0, c)

-- Define vectors AC and BC
def AC (c : ℝ) : ℝ × ℝ := ((C c).1 - A.1, (C c).2 - A.2)
def BC (c : ℝ) : ℝ × ℝ := ((C c).1 - B.1, (C c).2 - B.2)

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the centroid of the triangle
noncomputable def centroid (c : ℝ) : ℝ × ℝ := ((A.1 + B.1 + (C c).1) / 3, (A.2 + B.2 + (C c).2) / 3)

theorem triangle_abc_properties :
  ∀ c : ℝ, dot_product (AC c) (BC c) = 0 →
    (c = 2 ∨ c = -2) ∧
    ((c = 2 → centroid c = (1, 2/3)) ∧
     (c = -2 → centroid c = (1, -2/3))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1287_128736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_loss_percentage_l1287_128761

theorem watch_loss_percentage
  (cost_price : ℝ)
  (additional_amount : ℝ)
  (gain_percentage : ℝ)
  (h1 : cost_price = 933.33)
  (h2 : additional_amount = 140)
  (h3 : gain_percentage = 5) :
  let selling_price := cost_price + additional_amount - (cost_price * (1 + gain_percentage / 100))
  let loss_amount := cost_price - selling_price
  let loss_percentage := (loss_amount / cost_price) * 100
  loss_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_loss_percentage_l1287_128761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition1_condition2_capacity_A_is_3_capacity_B_is_4_total_capacity_l1287_128718

/-- Represents the capacity of a type A car in tons -/
def capacity_A : ℝ := 3

/-- Represents the capacity of a type B car in tons -/
def capacity_B : ℝ := 4

/-- The total capacity of 2 type A cars and 1 type B car is 10 tons -/
theorem condition1 : 2 * capacity_A + capacity_B = 10 := by
  simp [capacity_A, capacity_B]
  norm_num

/-- The total capacity of 1 type A car and 2 type B cars is 11 tons -/
theorem condition2 : capacity_A + 2 * capacity_B = 11 := by
  simp [capacity_A, capacity_B]
  norm_num

/-- The capacity of a type A car is 3 tons -/
theorem capacity_A_is_3 : capacity_A = 3 := by rfl

/-- The capacity of a type B car is 4 tons -/
theorem capacity_B_is_4 : capacity_B = 4 := by rfl

/-- The total capacity of 6 type A cars and 8 type B cars is 50 tons -/
theorem total_capacity : 6 * capacity_A + 8 * capacity_B = 50 := by
  simp [capacity_A, capacity_B]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition1_condition2_capacity_A_is_3_capacity_B_is_4_total_capacity_l1287_128718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_in_interval_l1287_128769

-- Define the function f(x) = 2^x + 3x - 6
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + 3*x - 6

-- Theorem stating that there exists a root of f in the interval [1, 2]
theorem f_has_root_in_interval :
  ∃ x ∈ Set.Icc 1 2, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_in_interval_l1287_128769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_2012_times_l1287_128705

-- Define the custom operation
def custom_op : ℕ → ℕ → ℕ := sorry

-- Define properties of the custom operation
axiom op_property_1 : custom_op 3 4 = 2
axiom op_property_2 : custom_op 1 2 = 2

-- Define a function that applies the custom operation n times to an initial value
def apply_n_times : ℕ → ℕ → ℕ
  | 0, initial => initial
  | n + 1, initial => custom_op (apply_n_times n initial) 2

-- State the theorem
theorem custom_op_2012_times :
  apply_n_times 2012 (custom_op 1 2) = 1 := by
  sorry

#check custom_op_2012_times

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_2012_times_l1287_128705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1287_128770

theorem relationship_abc (x : ℝ) (h : x > 2) : (1/3)^3 < Real.log x ∧ Real.log x < x^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1287_128770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_of_3_l1287_128788

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 3 else x^2 - 2*x

-- Theorem statement
theorem f_inverse_of_3 (m : ℝ) :
  f m = 3 → m = 0 ∨ m = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_of_3_l1287_128788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a10_difference_l1287_128701

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → (a (n + 1) - a n) ∈ Finset.range n

def max_a10 : ℕ := 512

def min_a10 : ℕ := 10

theorem a10_difference :
  ∃ a : ℕ → ℕ, is_valid_sequence a ∧ 
    (∀ b : ℕ → ℕ, is_valid_sequence b → a 10 ≤ max_a10) ∧
    (∀ b : ℕ → ℕ, is_valid_sequence b → a 10 ≥ min_a10) ∧
    max_a10 - min_a10 = 502 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a10_difference_l1287_128701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_divisibility_and_power_l1287_128796

/-- Sequence P(n) representing the number of valid colorings of a 2 × n grid -/
def P : ℕ → ℕ
  | 0 => 4  -- Added case for 0
  | 1 => 4
  | 2 => 15
  | n + 3 => 3 * (P (n + 2) + P (n + 1))

/-- The statement to be proved -/
theorem coloring_divisibility_and_power :
  (∃ k : ℕ, P 1989 = 3 * k) ∧
  (∃ m : ℕ, P 1989 = 3^994 * m ∧ ¬(∃ l : ℕ, P 1989 = 3^995 * l)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_divisibility_and_power_l1287_128796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_derivative_of_f_l1287_128703

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x * log (1 - 3 * x)

-- State the theorem
theorem fourth_derivative_of_f (x : ℝ) (h : x ≠ 1/3) : 
  (deriv^[4] f) x = -54 * (4 - 3*x) / (1 - 3*x)^4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_derivative_of_f_l1287_128703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_lateral_area_l1287_128755

/-- The lateral surface area of a cylinder formed by rotating a rectangle -/
noncomputable def lateral_surface_area (length width : ℝ) : ℝ :=
  2 * Real.pi * length * width

/-- Theorem: The lateral surface area of the specific cylinder is 16π -/
theorem cylinder_lateral_area :
  lateral_surface_area 4 2 = 16 * Real.pi := by
  unfold lateral_surface_area
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_lateral_area_l1287_128755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_of_expression_l1287_128795

theorem smallest_value_of_expression (a b c : ℤ) (ω : ℂ) : 
  (∃ n : ℤ, a = n - 1 ∧ b = n ∧ c = n + 1) →  -- sequential integers
  ω^4 = 1 →
  ω ≠ 1 →
  ω.im ≠ 0 →  -- ω is not purely real
  ∃ m : ℝ, m = Complex.abs (↑a + ↑b * ω + ↑c * ω^2) ∧ 
    (∀ k : ℝ, k = Complex.abs (↑a + ↑b * ω + ↑c * ω^2) → m ≤ k) ∧ 
    m = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_of_expression_l1287_128795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_10_primes_over_50_l1287_128739

def is_prime (n : ℕ) : Bool := sorry

def first_10_primes_over_50 : List ℕ := sorry

theorem sum_first_10_primes_over_50 : 
  (first_10_primes_over_50.filter (λ x => is_prime x && x > 50)).sum = 732 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_10_primes_over_50_l1287_128739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_sin_greater_than_one_l1287_128786

theorem negation_of_exists_sin_greater_than_one :
  (¬ ∃ x : ℝ, x ≥ π / 2 ∧ Real.sin x > 1) ↔ (∀ x : ℝ, x < π / 2 → Real.sin x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_sin_greater_than_one_l1287_128786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_first_player_wins_533_first_player_wins_1000_l1287_128768

-- Define the game state
structure GameState where
  sum : ℕ
  player : Bool -- True for player 1, False for player 2

-- Define the possible moves
inductive Move where
  | one : Move
  | two : Move

-- Define the game rules
def nextState (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.one => { sum := state.sum + 1, player := ¬state.player }
  | Move.two => { sum := state.sum + 2, player := ¬state.player }

-- Define the winning condition
def isWinningState (target : ℕ) (state : GameState) : Bool :=
  state.sum = target

-- Define the theorem for the winning strategy
theorem first_player_wins (target : ℕ) :
  ∃ (strategy : GameState → Move),
    ∀ (game : ℕ → GameState),
      game 0 = { sum := 0, player := true } →
      (∀ n, game (n + 1) = nextState (game n) (if (game n).player then strategy (game n) else if (game n).sum % 3 = 0 then Move.one else Move.two)) →
      ∃ n, isWinningState target (game n) ∧ ¬(game n).player :=
by sorry

-- Instantiate the theorem for target 533
theorem first_player_wins_533 :
  ∃ (strategy : GameState → Move),
    ∀ (game : ℕ → GameState),
      game 0 = { sum := 0, player := true } →
      (∀ n, game (n + 1) = nextState (game n) (if (game n).player then strategy (game n) else if (game n).sum % 3 = 0 then Move.one else Move.two)) →
      ∃ n, isWinningState 533 (game n) ∧ ¬(game n).player :=
by sorry

-- Instantiate the theorem for target 1000
theorem first_player_wins_1000 :
  ∃ (strategy : GameState → Move),
    ∀ (game : ℕ → GameState),
      game 0 = { sum := 0, player := true } →
      (∀ n, game (n + 1) = nextState (game n) (if (game n).player then strategy (game n) else if (game n).sum % 3 = 0 then Move.one else Move.two)) →
      ∃ n, isWinningState 1000 (game n) ∧ ¬(game n).player :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_first_player_wins_533_first_player_wins_1000_l1287_128768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crow_speed_l1287_128721

theorem crow_speed (nest_to_ditch : ℝ) (num_trips : ℕ) (total_time : ℝ) :
  nest_to_ditch = 250 →
  num_trips = 15 →
  total_time = 1.5 →
  (2 * nest_to_ditch * (num_trips : ℝ)) / (1000 * total_time) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crow_speed_l1287_128721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_real_inequality_l1287_128733

theorem positive_real_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1 / a + 1 / b = 1) :
  ∀ n : ℕ, (a + b)^(n : ℝ) - a^(n : ℝ) - b^(n : ℝ) ≥ 2^(2*n : ℝ) - 2^((n+1) : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_real_inequality_l1287_128733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_locus_l1287_128715

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the rotation function
noncomputable def rotate (p : Point) (center : Point) (angle : ℝ) : Point :=
  { x := center.x + (p.x - center.x) * Real.cos angle - (p.y - center.y) * Real.sin angle,
    y := center.y + (p.x - center.x) * Real.sin angle + (p.y - center.y) * Real.cos angle }

-- Define the locus
def locus (C : Point) (f f' : Line) : Prop :=
  (f.a * C.x + f.b * C.y + f.c = 0) ∨ (f'.a * C.x + f'.b * C.y + f'.c = 0)

-- Theorem statement
theorem equilateral_triangle_locus 
  (ABC : Triangle) 
  (e : Line) 
  (h1 : ABC.A.x = 0 ∧ ABC.A.y = 0)  -- A is fixed at origin
  (h2 : ∃ (t : ℝ), ABC.B = Point.mk (e.a * t) (e.b * t))  -- B moves along line e
  (h3 : (ABC.A.x - ABC.B.x)^2 + (ABC.A.y - ABC.B.y)^2 = 
        (ABC.B.x - ABC.C.x)^2 + (ABC.B.y - ABC.C.y)^2)  -- ABC is equilateral
  (h4 : (ABC.A.x - ABC.B.x)^2 + (ABC.A.y - ABC.B.y)^2 = 
        (ABC.C.x - ABC.A.x)^2 + (ABC.C.y - ABC.A.y)^2)  -- ABC is equilateral
  : ∃ (f f' : Line), 
    f = Line.mk (e.a * Real.cos (π/3) - e.b * Real.sin (π/3)) (e.a * Real.sin (π/3) + e.b * Real.cos (π/3)) e.c ∧
    f' = Line.mk (e.a * Real.cos (-π/3) - e.b * Real.sin (-π/3)) (e.a * Real.sin (-π/3) + e.b * Real.cos (-π/3)) e.c ∧
    locus ABC.C f f' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_locus_l1287_128715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_example_l1287_128716

/-- The area of a circular sector with radius r and central angle θ (in degrees) -/
noncomputable def sectorArea (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * Real.pi * r^2

/-- Theorem: The area of a circular sector with radius 12 meters and central angle 40° is 16π square meters -/
theorem sector_area_example : sectorArea 12 40 = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_example_l1287_128716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_inequality_l1287_128706

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x / Real.log x
def g (k : ℝ) (x : ℝ) : ℝ := k * (x - 1)

-- Define what it means for a line to be tangent to a curve
def IsTangentLine (l : ℝ → ℝ) (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  DifferentiableAt ℝ f x₀ ∧ (deriv f x₀ = (l x₀ - l x₀) / (x₀ - x₀))

-- State the theorem
theorem tangent_and_inequality (k : ℝ) :
  (∀ x : ℝ, x > 0 ∧ x ≠ 1 → ¬ IsTangentLine (g k) f x) ∧
  (∃ x : ℝ, x ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧ f x ≤ g k x + 1/2 → k ≥ 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_inequality_l1287_128706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_5_units_from_M_l1287_128777

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The point M -/
def M : ℝ × ℝ := (1, 3)

/-- The first point on the x-axis -/
def A1 : ℝ × ℝ := (-3, 0)

/-- The second point on the x-axis -/
def A2 : ℝ × ℝ := (5, 0)

/-- Theorem stating that A1 and A2 are 5 units away from M -/
theorem points_5_units_from_M :
  distance M.1 M.2 A1.1 A1.2 = 5 ∧ distance M.1 M.2 A2.1 A2.2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_5_units_from_M_l1287_128777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_intersecting_square_l1287_128756

/-- A unit square on a plane -/
structure UnitSquare where
  center : ℝ × ℝ

/-- The set of unit squares -/
def M : Set UnitSquare := sorry

/-- The distance between centers of any two squares in M is at most 2 -/
axiom centers_distance_bound (s1 s2 : UnitSquare) (h1 : s1 ∈ M) (h2 : s2 ∈ M) :
  Real.sqrt ((s1.center.1 - s2.center.1)^2 + (s1.center.2 - s2.center.2)^2) ≤ 2

/-- A point is within a unit square if its coordinates are within 0.5 of the square's center -/
def within_square (p : ℝ × ℝ) (s : UnitSquare) : Prop :=
  |p.1 - s.center.1| ≤ 0.5 ∧ |p.2 - s.center.2| ≤ 0.5

/-- The theorem to be proved -/
theorem exists_intersecting_square :
  ∃ (s : UnitSquare), s.center = (1, 1) ∧ ∀ m, m ∈ M → ∃ (p : ℝ × ℝ), within_square p s ∧ within_square p m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_intersecting_square_l1287_128756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_minus_B_l1287_128776

theorem cos_A_minus_B (A B : ℝ) : 
  Real.tan A = 12/5 → 
  Real.cos B = -3/5 → 
  A ∈ Set.Icc π (3*π/2) → 
  B ∈ Set.Icc π (3*π/2) → 
  Real.cos (A - B) = 63/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_minus_B_l1287_128776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_is_2_sqrt_6_l1287_128763

/-- Circle C in polar form -/
def circle_C (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos (θ + Real.pi/4)

/-- Line l in parametric form -/
def line_l (x y t : ℝ) : Prop :=
  x = Real.sqrt 2 * t ∧ y = Real.sqrt 2 * t + 4 * Real.sqrt 2

/-- Minimum length of tangent PA -/
noncomputable def min_tangent_length : ℝ := 2 * Real.sqrt 6

/-- Theorem stating the minimum length of tangent PA -/
theorem min_tangent_length_is_2_sqrt_6 :
  ∀ (ρ θ x y t : ℝ), circle_C ρ θ → line_l x y t →
  ∃ (PA : ℝ), PA ≥ min_tangent_length ∧
  (∀ (PA' : ℝ), PA' ≥ min_tangent_length → PA ≤ PA') :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_is_2_sqrt_6_l1287_128763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_2_or_5_not_10_l1287_128749

theorem multiples_of_2_or_5_not_10 (n : ℕ) : n = 1500 →
  (Finset.filter (λ x ↦ (x % 2 = 0 ∨ x % 5 = 0) ∧ x % 10 ≠ 0) (Finset.range (n + 1))).card = 900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_2_or_5_not_10_l1287_128749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1287_128731

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  abc_positive : a > 0 ∧ b > 0 ∧ c > 0
  angle_sum : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : Real.cos (2 * t.C) = -1/4)
  (h2 : t.a = 2)
  (h3 : 2 * Real.sin t.A = Real.sin t.C) :
  Real.sin t.C = Real.sqrt 10 / 4 ∧ 
  t.c = 4 ∧ 
  (t.b = Real.sqrt 6 ∨ t.b = 2 * Real.sqrt 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1287_128731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1287_128738

def A : Set ℚ := {-1, 0, 1, 2}
def B : Set ℚ := {x : ℚ | x^2 - x ≤ 0}

theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1287_128738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_debt_l1287_128764

theorem smallest_debt (cow_value sheep_value : ℕ) 
  (hcow : cow_value = 400) (hsheep : sheep_value = 250) : ℕ := by
  let smallestDebt : ℕ := 50
  have isSmallest : ∀ d : ℕ, d > 0 → (∃ c s : ℤ, d = cow_value * c + sheep_value * s) → d ≥ smallestDebt := by
    sorry
  have canBeSettled : ∃ c s : ℤ, smallestDebt = cow_value * c + sheep_value * s := by
    sorry
  exact smallestDebt

#check smallest_debt

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_debt_l1287_128764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_range_l1287_128772

open Real

theorem odd_function_range (f : ℝ → ℝ) (t : ℝ) :
  (∀ x, f (-x) = -f x) →
  ContinuousOn f (Set.Ioo (-π/2) (π/2)) →
  (∀ x ∈ Set.Ioo (-π/2) 0, 
    ∃ f', HasDerivAt f f' x ∧ f' * cos x - f x * sin x < 0) →
  f t * cos t < (1/2) * f (π/3) →
  t ∈ Set.Ioo (π/3) (π/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_range_l1287_128772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_tournament_matches_l1287_128720

theorem football_tournament_matches (teams : Finset ℕ) (matches_played : ℕ → ℕ) : 
  teams.card = 16 → 
  (∀ t, t ∈ teams → matches_played t ≤ 15) →
  ∃ t1 t2, t1 ∈ teams ∧ t2 ∈ teams ∧ t1 ≠ t2 ∧ matches_played t1 = matches_played t2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_tournament_matches_l1287_128720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eldest_sibling_age_is_34_point_5_l1287_128758

/-- Represents a family of siblings with specific age differences. -/
structure SiblingFamily where
  num_siblings : ℕ
  age_difference : ℕ
  combined_age : ℚ
  excluded_sibling : ℕ

/-- Calculates the age of the eldest sibling in the family. -/
noncomputable def eldest_sibling_age (family : SiblingFamily) : ℚ :=
  let youngest_age := family.combined_age / (family.num_siblings - 1 : ℚ) - 
    ((family.num_siblings - 1 : ℚ) * family.age_difference) / 2
  youngest_age + ((family.num_siblings - 1 : ℚ) * family.age_difference)

/-- Theorem stating the age of the eldest sibling in the given family. -/
theorem eldest_sibling_age_is_34_point_5 : 
  let family : SiblingFamily := {
    num_siblings := 9,
    age_difference := 4,
    combined_age := 140,
    excluded_sibling := 3
  }
  eldest_sibling_age family = 69/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eldest_sibling_age_is_34_point_5_l1287_128758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_correct_l1287_128702

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a circle in the polar coordinate system -/
structure PolarCircle where
  center : PolarPoint
  equation : ℝ → ℝ → Prop

/-- The circle passing through the pole with center at (√2, π) -/
noncomputable def specialCircle : PolarCircle :=
  { center := { r := Real.sqrt 2, θ := Real.pi },
    equation := fun ρ θ ↦ ρ = -2 * Real.sqrt 2 * Real.cos θ }

theorem special_circle_correct : 
  (specialCircle.equation 0 0) ∧ 
  (∃ (ρ θ : ℝ), specialCircle.equation ρ θ ∧ 
    ρ * Real.cos θ = -specialCircle.center.r ∧
    ρ * Real.sin θ = 0) := by
  sorry

#check special_circle_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_correct_l1287_128702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kyungsoo_second_place_l1287_128730

-- Define the students and their jump distances
noncomputable def kyungsoo_jump : ℝ := 2.3
noncomputable def younghee_jump : ℝ := 9/10
noncomputable def jinju_jump : ℝ := 1.8
noncomputable def chanho_jump : ℝ := 2.5

-- Define a function to check if a given distance is the second longest
def is_second_longest (x y z w : ℝ) : Prop :=
  (x < max y (max z w) ∧ x > min (max y z) (max y w) ∧ x > min (max z w) (max y w)) ∨
  (x = max y (max z w) ∧ x < max z w)

-- Theorem stating that Kyungsoo's jump is the second longest
theorem kyungsoo_second_place : 
  is_second_longest kyungsoo_jump chanho_jump jinju_jump younghee_jump := by
  sorry

#check kyungsoo_second_place

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kyungsoo_second_place_l1287_128730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1287_128735

noncomputable section

open Real

variable (x : ℝ)
variable (k : ℤ)

noncomputable def a : ℝ × ℝ := (sin (π * x / 2), sin (π / 3))
noncomputable def b : ℝ × ℝ := (cos (π * x / 2), cos (π / 3))

noncomputable def f (x : ℝ) : ℝ := sin (π * x / 2 - π / 3)

variable (A B C : ℝ)

theorem vector_problem (h1 : ∃ (t : ℝ), a = t • b)
                       (h2 : 0 < A ∧ A < B ∧ B < π)
                       (h3 : A + B + C = π)
                       (h4 : f (4 * A / π) = 1/2)
                       (h5 : f (4 * B / π) = 1/2) :
  (∃ (x : ℝ), sin (π * x / 2 - π / 3) = 0) ∧
  (∃ (k : ℤ), ∀ (x : ℝ), f x = f (5/3 + 2 * k - x)) ∧
  (Finset.sum (Finset.range 2013) (fun i => f (i + 1)) = 1/2) ∧
  (sin B / sin C = (sqrt 6 + sqrt 2) / 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1287_128735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_102_103_l1287_128746

def g (x : ℤ) : ℤ := x^2 - 2*x + 2023

theorem gcd_g_102_103 : Int.gcd (g 102) (g 103) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_102_103_l1287_128746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yz_circle_radius_l1287_128774

-- Define the sphere
def sphere_center : ℝ × ℝ × ℝ := (2, 4, -7)

-- Define the circle in the xy-plane
def xy_circle_center : ℝ × ℝ × ℝ := (2, 4, 0)
def xy_circle_radius : ℝ := 1

-- Define the circle in the yz-plane
def yz_circle_center : ℝ × ℝ × ℝ := (0, 4, -7)

-- Define the radius of the sphere
noncomputable def sphere_radius : ℝ := Real.sqrt 50

-- Theorem to prove
theorem yz_circle_radius : 
  let r := Real.sqrt (sphere_radius ^ 2 - (sphere_center.1 - yz_circle_center.1) ^ 2)
  r = Real.sqrt 46 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yz_circle_radius_l1287_128774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_non_union_sets_l1287_128783

/-- Given n ≥ 5 different sets, there exist at least ⌈√(2n)⌉ different sets 
    such that no set among these is the union of any two other sets. -/
theorem existence_of_non_union_sets (n : ℕ) (h : n ≥ 5) : 
  ∃ (r : ℕ) (S : Finset (Set α)), 
    r = ⌈Real.sqrt (2 * n)⌉ ∧ 
    S.card = r ∧ 
    (∀ A ∈ S, ∀ B ∈ S, ∀ C ∈ S, A ≠ B ∪ C) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_non_union_sets_l1287_128783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_third_is_zero_f_max_value_on_interval_l1287_128762

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x + π / 6) + cos (2 * x + π / 6)

-- Theorem for part (I)
theorem f_at_pi_third_is_zero : f (π / 3) = 0 := by sorry

-- Theorem for part (II)
theorem f_max_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-π / 3) (π / 6) ∧
  f x = 2 ∧
  ∀ y ∈ Set.Icc (-π / 3) (π / 6), f y ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_third_is_zero_f_max_value_on_interval_l1287_128762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pepperoni_coverage_half_l1287_128745

/-- Represents a circular pizza with pepperoni toppings -/
structure PepperoniPizza where
  pizza_diameter : ℝ
  pepperoni_count : ℕ
  pepperoni_across_diameter : ℕ

/-- Calculates the fraction of the pizza covered by pepperoni -/
noncomputable def pepperoni_coverage (p : PepperoniPizza) : ℝ :=
  let pizza_area := Real.pi * (p.pizza_diameter / 2) ^ 2
  let pepperoni_radius := p.pizza_diameter / (2 * p.pepperoni_across_diameter : ℝ)
  let pepperoni_area := Real.pi * pepperoni_radius ^ 2
  (p.pepperoni_count : ℝ) * pepperoni_area / pizza_area

theorem pepperoni_coverage_half (p : PepperoniPizza) 
  (h1 : p.pizza_diameter = 16)
  (h2 : p.pepperoni_count = 32)
  (h3 : p.pepperoni_across_diameter = 8) : 
  pepperoni_coverage p = 1 / 2 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pepperoni_coverage_half_l1287_128745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_condition_l1287_128790

-- Define the circle
def circle_eq (D E F : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + D*x + E*y + F = 0

-- Define tangency to x-axis
def tangent_to_x_axis (D E F : ℝ) : Prop :=
  ∃ (x : ℝ), circle_eq D E F x 0 ∧ ∀ (y : ℝ), y ≠ 0 → ¬(circle_eq D E F x y)

-- State the theorem
theorem circle_tangent_condition (D E F : ℝ) :
  (tangent_to_x_axis D E F → D^2 = 4*F) ∧
  ¬(D^2 = 4*F → tangent_to_x_axis D E F) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_condition_l1287_128790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_OA_perp_OB_l1287_128751

-- Define the ellipse parameters
noncomputable def a : ℝ := Real.sqrt 8
noncomputable def b : ℝ := 2

-- Define the points on the ellipse
noncomputable def M : ℝ × ℝ := (2, Real.sqrt 2)
noncomputable def N : ℝ × ℝ := (Real.sqrt 6, 1)
def O : ℝ × ℝ := (0, 0)

-- Define the line parameter
noncomputable def k : ℝ := Real.sqrt 5

-- Axioms
axiom h1 : a > b
axiom h2 : b > 0
axiom h3 : k > 0

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 8/3

-- Define the line equation
def line_eq (x y : ℝ) : Prop := y = k * x + 4

-- Theorem 1: The equation of ellipse E
theorem ellipse_equation : 
  (∀ x y, ellipse_eq x y ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  ellipse_eq M.1 M.2 ∧ ellipse_eq N.1 N.2 ∧ ellipse_eq O.1 O.2 :=
sorry

-- Define points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Axioms for A and B
axiom hA : ellipse_eq A.1 A.2 ∧ line_eq A.1 A.2
axiom hB : ellipse_eq B.1 B.2 ∧ line_eq B.1 B.2

-- Axiom for line tangent to circle
axiom h_tangent : ∃ x y, line_eq x y ∧ circle_eq x y ∧ 
  ∀ x' y', line_eq x' y' → circle_eq x' y' → (x, y) = (x', y')

-- Theorem 2: OA perpendicular to OB
theorem OA_perp_OB : A.1 * B.1 + A.2 * B.2 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_OA_perp_OB_l1287_128751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_parameter_range_l1287_128723

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1) - x^2

-- State the theorem
theorem function_inequality_implies_parameter_range (a : ℝ) :
  (∀ p q : ℝ, 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q →
    (f a (p + 1) - f a (q + 1)) / (p - q) > 1) →
  a ≥ 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_parameter_range_l1287_128723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_120_triangle_in_subdivision_l1287_128729

/-- A triangle represented by its three angles -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180
  all_positive : 0 < angle1 ∧ 0 < angle2 ∧ 0 < angle3

/-- A subdivision of a triangle into smaller triangles -/
structure Subdivision where
  initial : Triangle
  resulting : List Triangle
  is_subdivision : Prop -- Placeholder for subdivision property

/-- The main theorem statement -/
theorem exists_120_triangle_in_subdivision 
  (initial : Triangle) 
  (h_initial : initial.angle1 ≤ 120 ∧ initial.angle2 ≤ 120 ∧ initial.angle3 ≤ 120) 
  (sub : Subdivision) 
  (h_sub : sub.initial = initial) : 
  ∃ t ∈ sub.resulting, t.angle1 ≤ 120 ∧ t.angle2 ≤ 120 ∧ t.angle3 ≤ 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_120_triangle_in_subdivision_l1287_128729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1287_128747

open Real

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (sin α + cos α, sin α - cos α)

-- Define the line l in polar coordinates
def line_l (ρ θ : ℝ) : Prop := sqrt 2 * ρ * sin (π / 4 - θ) + 1 = 0

-- Theorem statement
theorem intersection_distance :
  ∃ (A B : ℝ × ℝ) (α₁ α₂ ρ₁ ρ₂ θ₁ θ₂ : ℝ),
    curve_C α₁ = A ∧
    curve_C α₂ = B ∧
    line_l ρ₁ θ₁ ∧
    line_l ρ₂ θ₂ ∧
    A ≠ B ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1287_128747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_xOz_plane_l1287_128771

/-- The point symmetric to (x, y, z) with respect to the xOz plane --/
def symmetric_point (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2.1, p.2.2)

/-- The xOz plane in a 3D Cartesian coordinate system --/
def xOz_plane (p : ℝ × ℝ × ℝ) : Prop :=
  p.2.1 = 0

theorem symmetry_xOz_plane :
  symmetric_point (-1, 2, 1) = (-1, -2, 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_xOz_plane_l1287_128771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_g_is_minimum_of_F_l1287_128734

/-- The quadratic function f(x) = x^2 + 2x + 1 -/
noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- The function F(x) = f(x) - kx -/
noncomputable def F (k : ℝ) (x : ℝ) : ℝ := f x - k*x

/-- The minimum value of F(x) over [-1, 1] as a function of k -/
noncomputable def g (k : ℝ) : ℝ :=
  if k ≤ 0 then k
  else if k < 4 then k - k^2/4
  else 4 - k

theorem f_properties :
  f 0 = 1 ∧ f (-1) = 0 ∧ ∀ x, x ≠ -1 → f x ≠ 0 := by sorry

theorem g_is_minimum_of_F :
  ∀ k, ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → F k x ≥ g k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_g_is_minimum_of_F_l1287_128734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_given_solution_set_l1287_128722

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / exp x - a / x

/-- Theorem stating the range of a given the conditions -/
theorem a_range_given_solution_set (a : ℝ) :
  (∃ m n : ℝ, ∀ x : ℝ, f a x ≥ 0 ↔ m ≤ x ∧ x ≤ n) →
  0 < a ∧ a < 1 / exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_given_solution_set_l1287_128722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_dhoni_stream_speed_l1287_128759

/-- The speed of a stream given boat speed and time ratio -/
theorem stream_speed (boat_speed : ℝ) (time_ratio : ℝ) 
  (h1 : boat_speed > 0)
  (h2 : time_ratio > 1) :
  let stream_speed := (time_ratio - 1) / (time_ratio + 1) * boat_speed
  (boat_speed - stream_speed) * time_ratio = boat_speed + stream_speed →
  stream_speed = boat_speed * (time_ratio - 1) / (time_ratio + 1) :=
by
  sorry

/-- The specific case for the given problem -/
theorem dhoni_stream_speed : 
  let boat_speed := (72 : ℝ)
  let time_ratio := (2 : ℝ)
  let stream_speed := (time_ratio - 1) / (time_ratio + 1) * boat_speed
  stream_speed = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_dhoni_stream_speed_l1287_128759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_range_l1287_128700

open Set

def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem set_operations_and_range :
  (∀ a : ℝ,
    (A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 10}) ∧
    ((Aᶜ) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10}) ∧
    ((A ∩ C a).Nonempty → a > 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_range_l1287_128700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_problem_l1287_128797

theorem circle_tangency_problem : 
  ∃ (count : ℕ), count = 3 ∧ 
    count = (Finset.filter (λ x : ℕ ↦ x < 144 ∧ x % 2 = 1 ∧ 144 % x = 0) (Finset.range 144)).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_problem_l1287_128797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_equals_one_two_l1287_128754

-- Define set A
def A : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}

-- Define set B (now as a subset of ℤ)
def B : Set ℤ := {x : ℤ | (x : ℝ)^2 - 3*(x : ℝ) < 0}

-- Theorem to prove
theorem A_intersect_B_equals_one_two : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_equals_one_two_l1287_128754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_isosceles_right_triangle_area_l1287_128704

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

/-- The coordinates of point A (a vertex on the major axis) -/
def point_A : ℝ × ℝ := (2, 0)

/-- Predicate for a point being on the ellipse -/
def on_ellipse (p : ℝ × ℝ) : Prop := ellipse_equation p.1 p.2

/-- Predicate for a triangle being isosceles and right-angled -/
def is_isosceles_right_triangle (a b c : ℝ × ℝ) : Prop :=
  (a.1 - b.1)^2 + (a.2 - b.2)^2 = (a.1 - c.1)^2 + (a.2 - c.2)^2 ∧
  (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0

/-- Predicate for a triangle being inscribed in the ellipse -/
def inscribed_in_ellipse (a b c : ℝ × ℝ) : Prop :=
  on_ellipse a ∧ on_ellipse b ∧ on_ellipse c

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

/-- The main theorem -/
theorem inscribed_isosceles_right_triangle_area :
  ∃ (b c : ℝ × ℝ),
    is_isosceles_right_triangle point_A b c ∧
    inscribed_in_ellipse point_A b c ∧
    triangle_area point_A b c = 8/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_isosceles_right_triangle_area_l1287_128704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l1287_128741

theorem solution_satisfies_equation :
  let y : ℚ := -47/24
  (1/8 : ℝ)^(3*y+9 : ℝ) = (32 : ℝ)^(3*y+4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l1287_128741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1287_128778

theorem calculation_proof : 
  4 * Real.sin (45 * Real.pi / 180) + (Real.sqrt 2 - Real.pi) ^ 0 - Real.sqrt 8 + (1 / 3) ^ (-2 : Int) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1287_128778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l1287_128713

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ (x - 3)^2 + (y - 3)^2 = 10}

-- Define points A and B
def point_A : ℝ × ℝ := (2, 0)
def point_B : ℝ × ℝ := (0, 4)

-- Define the line that contains the center of circle C
def center_line (x y : ℝ) : Prop := 2*x - y - 3 = 0

-- Define point T
def point_T : ℝ × ℝ := (-1, 0)

-- Define the line l
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ (x - 3*y + 1 = 0 ∨ 13*x - 9*y + 13 = 0)}

-- State the theorem
theorem circle_and_line_problem :
  (∀ p : ℝ × ℝ, p ∈ circle_C ↔ (∃ (x y : ℝ), p = (x, y) ∧ (x - 3)^2 + (y - 3)^2 = 10)) ∧
  (∃ (x y : ℝ), center_line x y ∧ (x, y) ∈ circle_C) ∧
  point_A ∈ circle_C ∧
  point_B ∈ circle_C ∧
  point_T ∈ line_l ∧
  (∃ (P Q : ℝ × ℝ), P ∈ circle_C ∧ Q ∈ circle_C ∧ P ∈ line_l ∧ Q ∈ line_l ∧
    let (cx, cy) := (3, 3);
    let (px, py) := P;
    let (qx, qy) := Q;
    (px - cx) * (qx - cx) + (py - cy) * (qy - cy) = -5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l1287_128713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_range_min_max_values_l1287_128757

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 4*x + 6

-- Theorem for part 1
theorem monotonic_decreasing_range (m : ℝ) :
  (∀ x y, x ∈ Set.Icc m (m + 1) → y ∈ Set.Icc m (m + 1) → x ≤ y → f x ≥ f y) ↔ m ≤ 1 :=
sorry

-- Theorem for part 2
theorem min_max_values (a b : ℝ) :
  a < b →
  (∀ x, x ∈ Set.Icc a b → f a ≤ f x) →
  (∀ x, x ∈ Set.Icc a b → f x ≤ f b) →
  f a = a →
  f b = b →
  a = 2 ∧ b = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_range_min_max_values_l1287_128757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1287_128725

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 5

-- Define the line passing through the center (3,5)
def line_eq (k : ℝ) (x y : ℝ) : Prop := y - 5 = k * (x - 3)

-- Define the intersection of the line with the y-axis
def y_axis_intersection (k : ℝ) : ℝ × ℝ := (0, 5 - 3*k)

-- Define the midpoint condition
def is_midpoint (A B P : ℝ × ℝ) : Prop :=
  A.1 = (B.1 + P.1) / 2 ∧ A.2 = (B.2 + P.2) / 2

theorem line_equation_proof (k : ℝ) (A B : ℝ × ℝ) :
  circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧
  line_eq k A.1 A.2 ∧ line_eq k B.1 B.2 ∧
  is_midpoint A B (y_axis_intersection k) →
  k = 2 ∨ k = -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1287_128725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_line_slope_l1287_128799

-- Define the line equation
def line_equation (x y a : ℝ) : Prop := x + Real.sqrt 3 * y - a = 0

-- Define the inclination angle
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan (-m)

-- Theorem statement
theorem line_inclination_angle (a : ℝ) :
  ∃ (θ : ℝ), θ = inclination_angle (1 / Real.sqrt 3) ∧ 
  θ = π * 5 / 6 ∧ 
  0 ≤ θ ∧ θ < π :=
by
  -- Proof goes here
  sorry

-- Additional helper theorem to show the slope of the line
theorem line_slope (a : ℝ) :
  ∃ (m : ℝ), m = -Real.sqrt 3 / 3 ∧
  ∀ (x y : ℝ), line_equation x y a → y = m * x + (a / Real.sqrt 3) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_line_slope_l1287_128799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_residue_system_l1287_128773

theorem permutation_residue_system (n : ℕ) : 
  (∃ p : Fin n → Fin n, Function.Bijective p ∧ 
    (∀ k : Fin n, ∃ i : Fin n, (p i + i : ℕ) % n = k) ∧
    (∀ k : Fin n, ∃ i : Fin n, (p i - i : ℤ) % n = k)) ↔ 
  Nat.Coprime n 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_residue_system_l1287_128773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_animals_count_l1287_128732

/-- Represents a national park with bear and fox populations -/
structure Park where
  blackBears : ℕ
  whiteBears : ℕ
  brownBears : ℕ
  foxes : ℕ

/-- Calculate the total number of animals in a park -/
def totalAnimals (p : Park) : ℕ := p.blackBears + p.whiteBears + p.brownBears + p.foxes

/-- Theorem stating the total number of animals across all parks -/
theorem total_animals_count (parkA parkB parkC : Park) 
  (hA1 : parkA.blackBears = 60)
  (hA2 : parkA.whiteBears = parkA.blackBears / 2)
  (hA3 : parkA.brownBears = parkA.blackBears + 40)
  (hA4 : parkA.foxes = (parkA.blackBears + parkA.whiteBears + parkA.brownBears) / 5)
  (hB1 : parkB.blackBears = 3 * parkA.blackBears)
  (hB2 : parkB.whiteBears = parkB.blackBears / 3)
  (hB3 : parkB.brownBears = parkB.blackBears + 70)
  (hB4 : parkB.foxes = (parkB.blackBears + parkB.whiteBears + parkB.brownBears) / 5)
  (hC1 : parkC.blackBears = 2 * parkB.blackBears)
  (hC2 : parkC.whiteBears = parkC.blackBears / 4)
  (hC3 : parkC.brownBears = parkC.blackBears + 100)
  (hC4 : parkC.foxes = (parkC.blackBears + parkC.whiteBears + parkC.brownBears) / 5)
  : totalAnimals parkA + totalAnimals parkB + totalAnimals parkC = 1908 := by
  sorry

#check total_animals_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_animals_count_l1287_128732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1287_128728

/-- Sum of a geometric series with first term a, common ratio r, and n terms -/
noncomputable def geometricSum (a r : ℝ) (n : ℕ) : ℝ := a * (r^n - 1) / (r - 1)

/-- The problem statement -/
theorem geometric_series_sum : geometricSum 4 4 8 = 87380 := by
  -- Unfold the definition of geometricSum
  unfold geometricSum
  -- Simplify the expression
  simp [pow_succ]
  -- Perform the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1287_128728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_six_digit_palindromes_l1287_128752

/-- A six-digit palindrome is a number of the form abcdcba where a ≠ 0 -/
def SixDigitPalindrome : Type := 
  { n : ℕ // 100000 ≤ n ∧ n ≤ 999999 ∧ ∃ a b c d : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    n = a * 100000 + b * 10000 + c * 1000 + d * 100 + c * 10 + b * 1 + a }

/-- SixDigitPalindrome is finite -/
instance : Fintype SixDigitPalindrome :=
  sorry

/-- The number of six-digit palindromes is 9000 -/
theorem count_six_digit_palindromes : Fintype.card SixDigitPalindrome = 9000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_six_digit_palindromes_l1287_128752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1287_128719

noncomputable def f (a b x : ℝ) : ℝ := -1/3 * x^3 + 2*a * x^2 - 3*a^2 * x + b

noncomputable def f_prime (a x : ℝ) : ℝ := -(x - 3*a) * (x - a)

theorem function_properties (a b : ℝ) (h : 0 < a ∧ a < 1) :
  -- 1. Monotonicity intervals
  (∀ x, a < x ∧ x < 3*a → (f_prime a x > 0)) ∧
  (∀ x, (x < a ∨ x > 3*a) → (f_prime a x < 0)) ∧
  -- 2. Maximum value
  (∀ x, f a b x ≤ f a b (3*a)) ∧
  -- 3. Minimum value
  (∀ x, f a b a ≤ f a b x) ∧
  f a b a = -4/3 * a^3 + b ∧
  -- 4. Range of a when |f'(x)| ≤ a for x ∈ [a+1, a+2]
  (∀ x, x ∈ Set.Icc (a+1) (a+2) → |f_prime a x| ≤ a) →
  a ∈ Set.Icc (4/5) 1 :=
by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1287_128719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l1287_128798

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the angle θ
noncomputable def angle_A (t : Triangle) : ℝ := sorry

-- Define the length of BC
noncomputable def length_BC (t : Triangle) : ℝ := sorry

-- Define the centroid
noncomputable def centroid (t : Triangle) : ℝ × ℝ := sorry

-- Define the midpoint of AB
noncomputable def midpoint_AB (t : Triangle) : ℝ × ℝ := sorry

-- Define the midpoint of AC
noncomputable def midpoint_AC (t : Triangle) : ℝ × ℝ := sorry

-- Define the condition of concyclicity
def are_concyclic (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

-- Main theorem
theorem triangle_existence (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π) :
  (∃! t : Triangle, 
    angle_A t = θ ∧ 
    length_BC t = 1 ∧ 
    are_concyclic t.A (centroid t) (midpoint_AB t) (midpoint_AC t)) ↔ 
  θ ≤ π / 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l1287_128798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_angle_inequality_l1287_128707

theorem acute_triangle_angle_inequality (α β γ : ℝ) 
  (acute_triangle : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 ∧ 0 < γ ∧ γ < π/2)
  (angle_sum : α + β + γ = π) : 
  Real.sin α + Real.sin β > Real.cos α + Real.cos β + Real.cos γ := by
  sorry

#check acute_triangle_angle_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_angle_inequality_l1287_128707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_example_l1287_128748

/-- The time (in seconds) it takes for a train to cross a platform -/
noncomputable def train_crossing_time (train_length platform_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem stating that a train 250 meters long crossing a 200-meter platform at 90 kmph takes 18 seconds -/
theorem train_crossing_example : train_crossing_time 250 200 90 = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_example_l1287_128748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sided_triangle_partition_l1287_128785

/-- A triangle with distinct sides -/
structure DistinctSidedTriangle where
  vertices : Fin 3 → ℝ × ℝ
  distinct_sides : ∀ i j : Fin 3, i ≠ j → 
    ‖vertices i - vertices j‖ ≠ ‖vertices j - vertices ((j + 1) % 3)‖

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  vertices : Fin 3 → ℝ × ℝ
  isosceles : ∃ i : Fin 3, 
    ‖vertices i - vertices ((i + 1) % 3)‖ = ‖vertices i - vertices ((i + 2) % 3)‖

/-- A partition of a triangle into isosceles triangles -/
def IsoscelesPartition (t : DistinctSidedTriangle) :=
  { partition : List IsoscelesTriangle // 
    ∀ p : ℝ × ℝ, p ∈ Set.range t.vertices → ∃ i : IsoscelesTriangle, i ∈ partition ∧ p ∈ Set.range i.vertices }

/-- Two isosceles triangles are congruent if they have the same side lengths -/
def CongruentIsosceles (t1 t2 : IsoscelesTriangle) :=
  ∃ (f : Fin 3 → Fin 3), ∀ i j : Fin 3, 
    ‖t1.vertices i - t1.vertices j‖ = ‖t2.vertices (f i) - t2.vertices (f j)‖

theorem distinct_sided_triangle_partition (t : DistinctSidedTriangle) :
  ∃ (partition : IsoscelesPartition t), 
    partition.val.length = 7 ∧ 
    ∃ (t1 t2 t3 : IsoscelesTriangle), 
      t1 ∈ partition.val ∧ t2 ∈ partition.val ∧ t3 ∈ partition.val ∧ 
      CongruentIsosceles t1 t2 ∧ CongruentIsosceles t2 t3 ∧ CongruentIsosceles t1 t3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sided_triangle_partition_l1287_128785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_experiment_results_l1287_128714

/-- Represents the state of a cylinder with gas --/
structure CylinderState where
  pressure : ℝ
  volume : ℝ
  height : ℝ

/-- Represents the properties of a gas --/
structure Gas where
  name : String
  isSaturated : Bool

/-- Represents the experimental setup --/
structure Experiment where
  initialState : CylinderState
  finalState : CylinderState
  gas : Gas
  temperature : ℝ
  pistonSpeed : ℝ
  duration : ℝ

def nitrogen : Gas := { name := "Nitrogen", isSaturated := false }
def waterVapor : Gas := { name := "Water Vapor", isSaturated := true }

noncomputable def initialState : CylinderState := {
  pressure := 0.5,
  volume := 2,
  height := 1
}

noncomputable def experimentSetup (g : Gas) : Experiment := {
  initialState := initialState,
  finalState := { -- To be calculated
    pressure := 0,
    volume := 0,
    height := 0
  },
  gas := g,
  temperature := 100,
  pistonSpeed := 10 / 60, -- 10 cm/min converted to m/min
  duration := 7.5
}

/-- Calculates the final state of the experiment --/
noncomputable def calculateFinalState (e : Experiment) : CylinderState := sorry

/-- Calculates the power of the external force applied to the piston --/
noncomputable def calculatePower (e : Experiment) : ℝ := sorry

/-- Calculates the work done by the external force over a given time interval --/
noncomputable def calculateWork (e : Experiment) (interval : ℝ) : ℝ := sorry

theorem experiment_results :
  let nitrogenExp := experimentSetup nitrogen
  let waterVaporExp := experimentSetup waterVapor
  let nitrogenFinalState := calculateFinalState nitrogenExp
  let waterVaporFinalState := calculateFinalState waterVaporExp
  let powerRatio := calculatePower nitrogenExp / calculatePower waterVaporExp
  
  (powerRatio = 2) ∧
  (∃ (interval : ℝ), interval = 0.5 ∧
    (calculateWork nitrogenExp interval > 15 ∨
     calculateWork waterVaporExp interval > 15)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_experiment_results_l1287_128714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l1287_128792

/-- Properties of an equilateral triangle -/
theorem equilateral_triangle_properties (a : ℝ) (h_pos : a > 0) :
  let height := a * Real.sqrt 3 / 2
  let inscribed_radius := a * Real.sqrt 3 / 6
  let circumscribed_radius := a * Real.sqrt 3 / 3
  (height = a * Real.sqrt 3 / 2) ∧
  (inscribed_radius = a * Real.sqrt 3 / 6) ∧
  (circumscribed_radius = a * Real.sqrt 3 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l1287_128792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_box_height_l1287_128708

/-- The minimum height of a rectangular box with square bases, where the height is 4 units greater
    than the side of the square bases, and with a surface area of at least 130 square units. -/
theorem min_box_height : ℝ := by
  -- Let x be the side length of the square base
  let x : ℝ → ℝ := fun side => side
  -- Define the height of the box
  let height : ℝ → ℝ := fun side => x side + 4
  -- Define the surface area of the box
  let surface_area : ℝ → ℝ := fun side => 2 * (x side)^2 + 4 * (x side) * (height side)
  
  -- We need to prove that 25/3 is the minimum height
  have h1 : ∀ side, surface_area side ≥ 130 → height side ≥ 25/3 := by
    intro side assumption
    -- Here we would normally prove this, but we'll use sorry for now
    sorry
  
  -- The minimum height is 25/3
  exact 25/3

#check min_box_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_box_height_l1287_128708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1287_128744

-- Define the parametric equation of line l
def line_l (t : ℝ) : ℝ × ℝ := (t, 1 + 2 * t)

-- Define the polar equation of circle C
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

-- Define the standard form of line l
def line_l_standard (x y : ℝ) : Prop := y - 2 * x - 1 = 0

-- Define the Cartesian equation of circle C
def circle_C_cartesian (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Theorem stating that line l intersects circle C
theorem line_intersects_circle : ∃ (x y : ℝ), line_l_standard x y ∧ circle_C_cartesian x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1287_128744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_baths_per_year_l1287_128794

def num_dogs : ℕ := 2
def num_cats : ℕ := 3
def num_birds : ℕ := 4

def dog_baths_per_month : ℕ := 2
def cat_baths_per_month : ℕ := 1
def bird_baths_per_month : ℚ := 1 / 4

def months_in_year : ℕ := 12

theorem total_baths_per_year :
  (num_dogs * dog_baths_per_month * months_in_year) +
  (num_cats * cat_baths_per_month * months_in_year) +
  (num_birds * (bird_baths_per_month * ↑months_in_year).floor) = 96 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_baths_per_year_l1287_128794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_locus_l1287_128724

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle centered at the origin -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- The constant area of all rectangles -/
def T : ℝ := 1  -- Assigning a default value, can be changed as needed

/-- Predicate for a point being a vertex of a rectangle with area T -/
def isVertex (p : Point) : Prop :=
  ∃ r : Rectangle, area r = T ∧ (p.x = r.width / 2 ∨ p.x = -r.width / 2) ∧ 
                              (p.y = r.height / 2 ∨ p.y = -r.height / 2)

/-- Theorem: The locus of vertices forms two hyperbolas -/
theorem vertex_locus : 
  ∀ p : Point, isVertex p ↔ p.x * p.y = T / 4 ∨ p.x * p.y = -T / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_locus_l1287_128724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_fence_poles_l1287_128787

noncomputable def trapezoid_poles (parallel_side1 parallel_side2 height pole_distance overlap_percent : ℝ) : ℕ :=
  let parallel_poles1 := ⌈parallel_side1 / pole_distance⌉
  let parallel_poles2 := ⌈parallel_side2 / pole_distance⌉
  let non_parallel_distance := pole_distance * (1 - overlap_percent / 100)
  let non_parallel_poles := ⌈height / non_parallel_distance⌉
  let total_poles := parallel_poles1 + parallel_poles2 + 2 * non_parallel_poles
  (total_poles - 4).toNat

theorem trapezoid_fence_poles :
  trapezoid_poles 30 50 40 5 25 = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_fence_poles_l1287_128787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_zero_to_e_l1287_128726

open Real

/-- The function f(x) = ln(x) / x -/
noncomputable def f (x : ℝ) : ℝ := (log x) / x

/-- The domain of f(x) is (0, +∞) -/
def f_domain : Set ℝ := {x | x > 0}

theorem f_increasing_on_zero_to_e :
  StrictMonoOn f (Set.Ioo 0 (exp 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_zero_to_e_l1287_128726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_free_all_cells_a_cannot_free_all_cells_b_l1287_128791

-- Define the grid and cell types
def Grid := ℕ → ℕ → Bool
def Cell := ℕ × ℕ

-- Define the marked cells
def marked_cells : List Cell := sorry

-- Define the movement rule
def can_move (g : Grid) (c : Cell) (to1 : Cell) (to2 : Cell) : Prop := sorry

-- Define the state of the grid after a move
def move (g : Grid) (c : Cell) (to1 : Cell) (to2 : Cell) : Grid := sorry

-- Define what it means for a cell to be free
def is_free (g : Grid) (c : Cell) : Prop := sorry

-- Define the initial state for part (a)
def initial_state_a : Grid := sorry

-- Define the initial state for part (b)
def initial_state_b : Grid := sorry

-- Define what it means to free all marked cells
def all_marked_cells_free (g : Grid) : Prop := sorry

-- Theorem for part (a)
theorem cannot_free_all_cells_a : 
  ¬∃ (moves : List (Cell × Cell × Cell)), 
    all_marked_cells_free (moves.foldl (λ g m => move g m.1 m.2.1 m.2.2) initial_state_a) :=
by sorry

-- Theorem for part (b)
theorem cannot_free_all_cells_b : 
  ¬∃ (moves : List (Cell × Cell × Cell)), 
    all_marked_cells_free (moves.foldl (λ g m => move g m.1 m.2.1 m.2.2) initial_state_b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_free_all_cells_a_cannot_free_all_cells_b_l1287_128791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_enjoyment_paradox_l1287_128737

theorem chess_enjoyment_paradox
  (total : ℕ)
  (enjoy_percent : ℚ)
  (admit_enjoy_percent : ℚ)
  (admit_not_enjoy_percent : ℚ)
  (h1 : enjoy_percent = 70 / 100)
  (h2 : admit_enjoy_percent = 75 / 100)
  (h3 : admit_not_enjoy_percent = 80 / 100)
  : (let enjoy := (total : ℚ) * enjoy_percent
     let not_enjoy := (total : ℚ) - enjoy
     let say_not_enjoy := (1 - admit_enjoy_percent) * enjoy + admit_not_enjoy_percent * not_enjoy
     let enjoy_but_say_not := (1 - admit_enjoy_percent) * enjoy
     enjoy_but_say_not / say_not_enjoy) = 35 / 83 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_enjoyment_paradox_l1287_128737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1287_128717

/-- The time taken for a train to cross a platform -/
theorem train_crossing_time (train_length : ℝ) (platform_length : ℝ) (train_speed_kmph : ℝ) 
  (h1 : train_length = 350.048)
  (h2 : platform_length = 250)
  (h3 : train_speed_kmph = 72) : 
  (train_length + platform_length) / (train_speed_kmph * (1000 / 3600)) = 30.0024 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1287_128717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_approximately_11_24_l1287_128711

/-- Represents a rectangle with circles at its corners -/
structure RectangleWithCircles where
  width : ℝ
  height : ℝ
  radius_E : ℝ
  radius_F : ℝ
  radius_G : ℝ
  radius_H : ℝ

/-- Calculates the area inside the rectangle but outside the quarter circles -/
noncomputable def areaOutsideCircles (r : RectangleWithCircles) : ℝ :=
  r.width * r.height - Real.pi / 4 * (r.radius_E^2 + r.radius_F^2 + r.radius_G^2 + r.radius_H^2)

/-- The specific rectangle with circles from the problem -/
def problemRectangle : RectangleWithCircles :=
  { width := 4
    height := 6
    radius_E := 2
    radius_F := 3
    radius_G := 1
    radius_H := 1.5 }

theorem area_approximately_11_24 :
    ∃ ε > 0, abs (areaOutsideCircles problemRectangle - 11.24) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_approximately_11_24_l1287_128711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_tan_75_minus_3tan_15_tan_complement_tan_double_angle_l1287_128789

noncomputable def degree_to_radian (d : ℝ) : ℝ := d * Real.pi / 180

theorem arctan_tan_75_minus_3tan_15 :
  let f (x : ℝ) := degree_to_radian x
  ∃ y, 0 ≤ y ∧ y ≤ Real.pi ∧ 
    Real.arctan (Real.tan (f 75) - 3 * Real.tan (f 15)) = f 75 :=
by
  sorry

-- Additional definitions to represent the given conditions
theorem tan_complement (x : ℝ) : 
  Real.tan (Real.pi / 2 - x) = 1 / Real.tan x :=
by
  sorry

theorem tan_double_angle (x : ℝ) : 
  Real.tan (2 * x) = (2 * Real.tan x) / (1 - Real.tan x ^ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_tan_75_minus_3tan_15_tan_complement_tan_double_angle_l1287_128789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_line_chart_comparison_l1287_128765

/-- A double line chart is convenient for comparing changes in quantities between two data sets. -/
theorem double_line_chart_comparison : True := by
  -- This theorem doesn't require a formal proof, as it's a statement about data visualization
  trivial

/-- Double line charts allow for easy comparison of trends. -/
def compare_trends (data_set1 data_set2 : List Float) : Bool :=
  -- In a real implementation, this function would compare the trends
  -- For now, we'll just return true as a placeholder
  true

#eval compare_trends [1, 2, 3] [2, 3, 4]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_line_chart_comparison_l1287_128765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_optimal_ratio_l1287_128766

/-- Cylinder properties -/
structure Cylinder where
  R : ℝ  -- radius
  H : ℝ  -- height
  V : ℝ  -- volume
  volume_eq : V = Real.pi * R^2 * H
  R_pos : R > 0
  H_pos : H > 0
  V_pos : V > 0

/-- Surface area of a cylinder -/
noncomputable def surface_area (c : Cylinder) : ℝ :=
  2 * Real.pi * c.R^2 + 2 * Real.pi * c.R * c.H

/-- Theorem: For a cylinder with fixed volume, the ratio H/R equals 2 when the surface area is minimized -/
theorem cylinder_optimal_ratio (cyl : Cylinder) : 
  (∀ c : Cylinder, c.V = cyl.V → surface_area c ≥ surface_area cyl) → 
  cyl.H / cyl.R = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_optimal_ratio_l1287_128766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_equation_l1287_128712

open Real

theorem smallest_positive_solution_tan_equation :
  ∃ (x : ℝ), x > 0 ∧ x = π / 18 ∧
  tan (4 * x) + tan (5 * x) = 1 / cos (5 * x) ∧
  ∀ (y : ℝ), y > 0 ∧ tan (4 * y) + tan (5 * y) = 1 / cos (5 * y) → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_equation_l1287_128712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1287_128779

theorem sin_alpha_value (α β : Real) 
  (h1 : 0 < α) (h2 : α < π / 2) (h3 : π / 2 < β) (h4 : β < π)
  (h5 : Real.cos β = -1/3) (h6 : Real.sin (α + β) = 7/9) : Real.sin α = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1287_128779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_triangle_from_triangle_l1287_128753

-- Define a structure for a triangle
structure Triangle (α : Type) [LinearOrderedField α] where
  a : α
  b : α
  c : α
  -- Triangle inequality conditions
  ab_gt_c : a + b > c
  ac_gt_b : a + c > b
  bc_gt_a : b + c > a

-- Theorem statement
theorem sqrt_triangle_from_triangle {α : Type} [LinearOrderedField α] 
  (t : Triangle ℝ) : 
  Triangle ℝ where
  a := Real.sqrt t.a
  b := Real.sqrt t.b
  c := Real.sqrt t.c
  ab_gt_c := by sorry
  ac_gt_b := by sorry
  bc_gt_a := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_triangle_from_triangle_l1287_128753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_12345_properties_l1287_128709

-- Define a and b
noncomputable def a : ℝ := 3 + 2 * Real.sqrt 2
noncomputable def b : ℝ := 3 - 2 * Real.sqrt 2

-- Define the sequence R_n
noncomputable def R (n : ℕ) : ℝ := (1 / 2) * (a ^ n + b ^ n)

-- State the theorem
theorem R_12345_properties :
  ∃ (m : ℤ), (R 12345 = m) ∧ (m % 10 = 9) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_12345_properties_l1287_128709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_gain_l1287_128743

-- Define the false weight used by the shopkeeper
noncomputable def false_weight : ℝ := 950

-- Define the true weight (1 kg in grams)
noncomputable def true_weight : ℝ := 1000

-- Define the percentage gain calculation
noncomputable def percentage_gain (false_weight true_weight : ℝ) : ℝ :=
  ((true_weight - false_weight) / true_weight) * 100

-- Theorem statement
theorem shopkeeper_gain :
  percentage_gain false_weight true_weight = 5 := by
  -- Unfold the definition of percentage_gain
  unfold percentage_gain
  -- Simplify the expression
  simp [false_weight, true_weight]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_gain_l1287_128743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_ST_l1287_128742

-- Define the points
variable (P Q R S T : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
axiom side_length : 
  ‖P - Q‖ = 6 ∧ ‖Q - R‖ = 6 ∧ ‖R - S‖ = 6 ∧ ‖S - P‖ = 6 ∧ ‖S - Q‖ = 6

axiom diagonal_length : ‖P - T‖ = 14 ∧ ‖R - T‖ = 14

-- Theorem to prove
theorem length_ST : ‖S - T‖ = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_ST_l1287_128742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_solution_sum_l1287_128784

theorem positive_solution_sum (a b : ℕ) (h : (Real.sqrt (a : ℝ) - b : ℝ)^2 + 10*(Real.sqrt (a : ℝ) - b : ℝ) = 40) : a + b = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_solution_sum_l1287_128784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_equilateral_l1287_128782

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Angles A, B, C form an arithmetic sequence -/
def is_arithmetic_sequence (t : Triangle) : Prop :=
  2 * t.B = t.A + t.C

/-- Vector m = (sin(A/2), cos(A/2)) -/
noncomputable def m (t : Triangle) : Fin 2 → ℝ
  | 0 => Real.sin (t.A / 2)
  | 1 => Real.cos (t.A / 2)
  | _ => 0

/-- Vector n = (cos(A/2), -√3*cos(A/2)) -/
noncomputable def n (t : Triangle) : Fin 2 → ℝ
  | 0 => Real.cos (t.A / 2)
  | 1 => -Real.sqrt 3 * Real.cos (t.A / 2)
  | _ => 0

/-- f(A) = m · n -/
noncomputable def f (t : Triangle) : ℝ :=
  (m t 0) * (n t 0) + (m t 1) * (n t 1)

/-- Main theorem -/
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : is_arithmetic_sequence t) 
  (h2 : f t = -Real.sqrt 3 / 2) : 
  t.A = t.B ∧ t.B = t.C ∧ t.A = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_equilateral_l1287_128782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_is_pi_l1287_128710

/-- A configuration of three intersecting semicircles -/
structure SemicircleConfiguration where
  /-- The radius of each semicircle -/
  radius : ℝ
  /-- Predicate ensuring the configuration is valid -/
  is_valid : radius > 0

/-- The area of the intersection region in the semicircle configuration -/
noncomputable def intersection_area (config : SemicircleConfiguration) : ℝ :=
  Real.pi * config.radius^2 / 2

/-- Theorem stating that for a semicircle configuration with radius 2,
    the intersection area is equal to π -/
theorem intersection_area_is_pi (config : SemicircleConfiguration)
    (h : config.radius = 2) :
    intersection_area config = Real.pi := by
  sorry

#check intersection_area_is_pi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_is_pi_l1287_128710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1287_128767

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∀ (x y : ℝ), -Real.pi/8 ≤ x ∧ x < y ∧ y ≤ 3*Real.pi/8 → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1287_128767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_pairs_l1287_128760

theorem arithmetic_progression_pairs : 
  ∃! (pairs : Finset (ℚ × ℚ)), 
    (∀ (x y : ℚ), (x, y) ∈ pairs ↔ 
      (x - 12 = y - x ∧ xy - y = y - x)) ∧ 
    (Finset.card pairs = 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_pairs_l1287_128760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_equality_test_l1287_128781

/-- F-distribution critical value for α = 0.05, df1 = 9, df2 = 13 -/
noncomputable def F_critical : ℝ := 2.72

/-- Test statistic for comparing two sample variances -/
noncomputable def F_statistic (s1_sq s2_sq : ℝ) : ℝ := s2_sq / s1_sq

/-- Hypothesis test for equality of population variances -/
theorem variance_equality_test (n1 n2 : ℕ) (s_X_sq s_Y_sq : ℝ) (α : ℝ) :
  n1 = 14 →
  n2 = 10 →
  s_X_sq = 0.84 →
  s_Y_sq = 2.52 →
  α = 0.1 →
  F_statistic s_X_sq s_Y_sq > F_critical := by
  intro h1 h2 h3 h4 h5
  have h6 : F_statistic s_X_sq s_Y_sq = 3 := by
    rw [F_statistic, h3, h4]
    norm_num
  rw [h6]
  norm_num
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_equality_test_l1287_128781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_19_divisibility_l1287_128793

/-- The base of the numeral system -/
def B : ℕ := sorry

/-- Condition that B is at least 10 -/
axiom B_ge_10 : B ≥ 10

/-- Definition of 792 in base B -/
def num_792 : ℕ := 7 * B^2 + 9 * B + 2

/-- Definition of 297 in base B -/
def num_297 : ℕ := 2 * B^2 + 9 * B + 7

/-- Divisibility condition -/
def is_divisible : Prop := ∃ k : ℕ, num_792 = k * num_297

/-- Theorem stating that 19 is the only base B ≥ 10 where 792 is divisible by 297 -/
theorem base_19_divisibility : 
  (B = 19 ↔ is_divisible) ∧ (∀ b : ℕ, b ≥ 10 ∧ b ≠ 19 → ¬(∃ k : ℕ, 7 * b^2 + 9 * b + 2 = k * (2 * b^2 + 9 * b + 7))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_19_divisibility_l1287_128793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_parallelepiped_volume_theorem_l1287_128780

/-- The volume of the set of points inside or within two units of a rectangular parallelepiped -/
noncomputable def extended_parallelepiped_volume (length width height extension : ℝ) : ℝ :=
  -- Original parallelepiped volume
  (length * width * height) +
  -- Extended volumes
  2 * (length * width * extension) +
  2 * (length * extension * height) +
  2 * (width * extension * height) +
  -- Quarter-cylinders volume
  Real.pi * extension^2 * (length + width + height) +
  -- Sphere octants volume
  4/3 * Real.pi * extension^3

/-- The theorem stating the volume of the extended parallelepiped -/
theorem extended_parallelepiped_volume_theorem :
  extended_parallelepiped_volume 2 3 6 2 = (540 + 164 * Real.pi) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_parallelepiped_volume_theorem_l1287_128780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_320_meters_l1287_128775

-- Define the train's speed in kmph
def train_speed_kmph : ℚ := 36

-- Define the time to cross an electric pole in seconds
def time_cross_pole : ℚ := 12

-- Define the time to cross the platform in seconds
def time_cross_platform : ℚ := 44

-- Convert kmph to m/s
def speed_ms (v : ℚ) : ℚ := v * (1000 / 3600)

-- Calculate the length of the train
def train_length (v : ℚ) (t : ℚ) : ℚ := speed_ms v * t

-- Calculate the total distance covered when crossing the platform
def total_distance (v : ℚ) (t : ℚ) : ℚ := speed_ms v * t

-- Calculate the length of the platform
def platform_length (v : ℚ) (t_pole t_platform : ℚ) : ℚ :=
  total_distance v t_platform - train_length v t_pole

-- Theorem statement
theorem platform_length_is_320_meters :
  platform_length train_speed_kmph time_cross_pole time_cross_platform = 320 := by
  -- Unfold definitions
  unfold platform_length
  unfold total_distance
  unfold train_length
  unfold speed_ms
  -- Perform algebraic simplifications
  simp [train_speed_kmph, time_cross_pole, time_cross_platform]
  -- The proof steps would go here, but we'll use sorry to skip the detailed proof
  sorry

#eval platform_length train_speed_kmph time_cross_pole time_cross_platform

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_320_meters_l1287_128775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l1287_128740

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x + y + 5 = 0

-- Define the distance function
noncomputable def distance (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

-- Theorem statement
theorem min_distance_to_origin :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 5 ∧
  ∀ (x y : ℝ), line_equation x y → distance x y ≥ min_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l1287_128740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1287_128750

-- Define the triangle ABC
def A : ℝ × ℝ := (0, -2)
def B : ℝ × ℝ := (4, -3)
def C : ℝ × ℝ := (3, 1)

-- Define the median line l₂
def median_l2 (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the area of the triangle
noncomputable def triangle_area : ℝ := 15 / 2

-- Theorem statement
theorem triangle_properties :
  (∀ x y : ℝ, median_l2 x y ↔ x + y - 1 = 0) ∧
  triangle_area = 15 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1287_128750
